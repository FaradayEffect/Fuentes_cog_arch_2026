# Lab 1: ACT-R agent + local LLM + ROS 2 / Gazebo

Pedro Fuentes, course "Cognitive Architectures and Their Applications".
(Русская версия: [README_ru.md](README_ru.md))

## What I built

A TurtleBot3 robot works in a small Gazebo world: two rooms (A and B), a hall between them and a
restricted zone where the robot must never go. There are two cubes. You give the robot a mission as
normal text, in Russian or English, for example "Отвези кубик 0 и кубик 1 в комнату А" or
"Bring cube 1 to room B".

- A small LLM that runs on my laptop (qwen2.5 3B in Ollama) reads the mission and writes a plan:
  navigate, pick, place.
- My ACT-R part checks that plan and decides, step by step, what is really executed.
- Nav2 drives the robot to the points that ACT-R gives it.
- A small forklift on the robot picks up the cube (every cube stands on a pallet) and puts it down.

The main idea in one line: **the LLM proposes, ACT-R decides.** The LLM never sends a command to the
robot. It can only write its plan into one buffer (`proposal`), and from there the plan has to pass the
validator, the production rules and the safety gate.

![The world: room A, the hall, room B and the restricted zone](docs/world_preview.png)

## How it works

```
mission text
     |
     v
   LLM  ------> buffer "proposal"   (a plan that nobody trusts yet)
     |
     v
 validator: marks the proposal valid or not
     |
     v  (rule ACCEPT-PLAN)
 buffer "plan"
     |
     v
 ACT-R productions choose the next step ---> buffer "action" (not approved yet)
     |
     v
 safety gate: approve or deny
     | approved
     v
 executor ---> Nav2 (driving) / forklift (pick, place) ---> robot in Gazebo
     ^                                                       |
     +------------ buffer "percept" <------------------------+
                   (robot pose, cubes, localization ok?)
```

If the gate denies a step or a step fails, recovery rules compete by utility: retry, try another
waypoint, skip the step, or ask the LLM for a new plan. If the robot ends up in the restricted zone or
loses its localization while driving, an emergency stop fires.

## The ACT-R part

I wrote a simplified ACT-R in Python (no external library), so I could see and test every piece.

| ACT-R idea | In my system |
|---|---|
| Modules | perception, memory, LLM, validator, gate, executor, evaluator |
| Buffers | 8: `goal`, `plan`, `percept`, `retrieval`, `proposal`, `action`, `motor`, `evaluation` |
| Chunks | immutable facts, e.g. `Chunk("action", {"cmd": "pick", "target": "cube_0", "approved": True})` |
| Productions | "if ... then ..." rules with a name and a utility: `ASK-LLM`, `SAFETY-GATE`, `EXECUTE`, `RECOVER-RETRY` ... |
| Conflict resolution | one production fires per cycle; if several match, the highest utility wins |
| Utility learning | `U <- U + a(R - U)`, a = 0.2; R = +1 after a success, -1 after a failure or a refusal |
| Declarative memory | episodes like "waypoint 0 of room A did not work"; activation `B = ln sum (t - tj)^-d`, d = 0.5, threshold -3.0 |
| Cycle | 0.1 s of simulation time |

Where to look in the code (`src/actr_lab1/actr_lab1/core/`):

- `buffers.py`, line 32: who can read and write each buffer (`DEFAULT_POLICY`)
- `productions.py`, line 92: the cycle (match, pick one rule, fire it); utility learning at line 32
- `agent.py`, line 769: the list of all production rules
- `memory.py`, line 53: activation of the declarative memory
- `gate.py`, line 36: the safety checks
- `planners.py`: the prompt for the LLM (`build_messages`) and the JSON schema of the plan (`PLAN_SCHEMA`)

## Access rights to the buffers

This is how ACT-R "controls access" here. Each module can read and write only its own buffers. If a
module tries something else, the buffer system raises `AccessDenied` and writes it to an audit log.

| Module | Reads | Writes |
|---|---|---|
| perception | - | `percept` |
| memory | `goal`, `percept`, `action` | `retrieval` |
| llm | `goal`, `percept`, `retrieval`, `evaluation` | only `proposal` |
| validator | `proposal`, `percept`, `goal` | `proposal` |
| gate | `action`, `percept`, `goal`, `plan` | `action` (the `approved` field) |
| executor | `action`, `motor` | `motor` |
| evaluator | `motor`, `action`, `goal`, `plan`, `percept` | `evaluation` |
| procedural (the productions) | all | `goal`, `plan`, `action`, `proposal`, `retrieval`, `motor` |

The executor only takes actions with `approved = True`. To be clear: this is a protection inside one
program, not an operating system sandbox.

## Safety gate

Every action goes through the gate before it reaches the robot:

- only `navigate`, `pick` and `place` are allowed (`stop` is never refused);
- no driving into the restricted zone and no picking a cube that lies there;
- no driving while the robot has lost its localization;
- `pick` only close to the cube (up to about 1 m; 1.5 m with the forklift, which drives the last part
  itself) and with an empty gripper;
- `place` only inside the right room and only if there is free floor;
- at most 200 actions per mission.

The emergency stop has utility 100, so it always wins. It fires if the robot is in the restricted zone,
or if it has had no localization for more than 2 s while driving. It cancels the Nav2 goal and sends
zero speed.

While the forklift picks up or puts down a pallet (20-40 s), the ACT-R cycle waits for it, so the
emergency stop rule does not run then. The forklift code has its own checks for these short drives: it
never comes closer than 20 cm to the restricted zone, it checks walls and other pallets, and it has a
time limit (60 s).

Example from my pure-Python simulation: the mission asks to go to the restricted area first. The gate
refuses that step (orange square), the recovery rule skips it (triangle) and the rest of the mission is
done. The other scenarios are in `docs/`.

![Safety demo: the step to the restricted zone is refused](docs/demo_forbidden.png)

## The forklift (new in this version)

TurtleBot3 Burger has no arm. In the first version the "gripper" was virtual: Gazebo just moved the
cube on top of the robot. Now the robot has a small **forklift** in front, and every cube stands on its
own small **pallet**, like in a warehouse:

- two flat **forks** (orange). They are thin at the tip (a wedge in 4 steps, from 4 mm to 15 mm) and
  thicker at the back, so they slide into the openings of the pallet easily;
- a **lift** (a prismatic joint) that moves the forks 4 cm up and down;
- a **counterweight** (250 g) inside the robot at the back, so it does not tip forward with a load;
- the **pallets** are 24 x 24 cm: 9 blocks under a deck, so the forks can go in from all 4 sides.
  An opening is 6.75 cm wide and 4.5 cm high; a fork is 3 cm wide and 1.5 cm thick, so there is
  about 1.9 cm of free space on each side of a fork and 1.8 cm above it.

The cubes are 22 cm again, like in the first version, so the lidar (18 cm above the floor) sees them.
The forklift itself stays under the lidar. With the forks the robot is about 42 cm long.

The forklift is added to the TurtleBot3 model when the simulation starts (`ros/arm_model.py`). One
standard plugin of `gazebo_ros_pkgs` moves the lift, so nothing has to be compiled:
`libgazebo_ros_joint_pose_trajectory.so` (topic `/arm/set_joint_trajectory`). There is no hidden force:
the pallet stays on the forks only because of its weight and friction.

Pick, step by step (`ros/ros_robot.py`, `_pick_with_arm`, line 841):

1. Nav2 brings the robot near the cube (this is the `navigate` step of the plan), about 0.8 m away.
2. The robot chooses the side of the pallet that is closest and free (no wall and no other cube in the
   way), drives to a point 75 cm from the centre of the pallet (about 63 cm from its edge) and turns
   to face it.
3. With the forks down it drives in slowly (4 cm/s) on the centre line of the pallet. If the forks are
   more than 1 cm off the line, or turned more than 1.5 degrees, when the tips reach the pallet, the
   robot stops before touching it and backs out.
4. When the forks are all the way in, it checks that the pallet has not moved (the forks did not push
   it), lifts, and checks in Gazebo that the pallet really came up (about 2 cm). If a check fails, the
   pick fails and the recovery rules of ACT-R decide what to do (retry, re-plan ...). After a lift the
   Nav2 costmaps are cleared around the robot (0.9 m square), so that old marks of that cube do not stay
   where the robot stands; the marks of the other cubes stay.

Place: the robot turns to the free spot that the agent chose, drives until the pallet is above it, puts
the forks down, backs out 40 cm and checks that the pallet stands on the floor. When the agent looks for
a free spot, it asks the forklift about each one (`drop_ok`): the way of the load there must be free.
If the load would sweep into another pallet while the robot turns, it first backs up a little, straight.

Some choices I had to make:

- **Forks instead of fingers.** Fingers need a gripper controller and contact physics, and in Gazebo
  Classic grasped objects often slip. On forks the load just lies on top, which is simple and stable,
  and it is how real warehouses do it.
- **The lidar sees the cubes, but it must not see its own load.** A cube on the forks is 13-17 cm in
  front of the lidar (the forks and the pallet are lower than the lidar). With the forklift, AMCL and
  the Nav2 costmaps skip lidar points closer than 25 cm (`laser_min_range`, `obstacle_min_range`), so
  the robot does not take its own load for an obstacle. Without the forklift (`arm:=false`) the Nav2
  parameters are the ones of the first version.
- **Nav2 knows that the robot is bigger:** robot radius 0.38 m (robot, forks and load are 35 cm, plus a
  little because the lidar sees a cube but not the pallet under it) instead of 0.12 m, and inflation
  radius 0.55 m instead of 0.40 m.
- **The robot stops further from a cube** (0.75-0.85 m instead of 0.55-0.65 m), so that it can turn
  without touching the pallet with the forks. The agent counts the robot as "next to the cube" up to
  1.3 m, and the forklift drives the last part by itself.
- **The last centimetres use the true pose from Gazebo.** AMCL is good for driving, but not for putting
  a 3 cm fork into a 6.75 cm opening. This is a simulation shortcut; a real robot would need a camera
  or another sensor for this part.
- **I checked the sizes with the real collision boxes:** the boxes of the forks (from `arm_model.py`)
  against the boxes of the pallet (from `worlds/rooms.world`), with the robot 1.5 cm to the side and
  turned 1.5 degrees, from the entry point until the forks are all the way in. Nothing touches
  (test `test_the_forks_go_in_without_touching_the_wood`).

For ACT-R the forklift is just part of the motor module: the productions decide *when* to pick or
place, and the forklift code does *how*. The gate still checks every pick and place before it starts.

If something is wrong with the forklift, the old virtual gripper is still there:
`sim.launch.py arm:=false` (robot without the forklift) or `agent.launch.py gripper:=virtual`.
By default (`gripper:=auto`) the agent checks if the robot in Gazebo has the forklift and prints
`gripper: the forklift (forks + lift)` or `gripper: virtual (Gazebo moves the cube)`.

## How many tokens the LLM uses

One plan is one LLM call:

- **in:** about 500 tokens. Most of it is the system prompt (the rules, the names of places and
  cubes, two examples), and the rest is the mission and the current state as JSON;
- **out:** about 50 tokens for a one-cube plan (4 steps), about 100 for two cubes (8 steps).

Usually one call per mission is enough. If the plan is wrong, the validator sends the errors back and
the LLM tries again (at most 3 answers, then a simple rule-based planner takes over). The context
window is set to 4096 tokens (`num_ctx`), so it fits easily.

The real numbers come from Ollama itself. Every run saves them in `events.jsonl`, and with the launch
argument `show_tokens:=true` the agent also prints them, for example
`ollama answered in 1.9 s, tokens: N in + M out`. By default this is off, so the normal log stays short.

Tokens and chunks are not the same thing. A token is a piece of text for the LLM (a word or a part of
a word). A chunk is a fact in ACT-R memory with a type and named slots, like `action(cmd=pick,
target=cube_0)`. The LLM works with tokens; ACT-R works with chunks.

## How to run it

### Docker (ROS 2 Humble + Gazebo + Nav2 + Ollama), Windows 11 with WSL2 or Linux

Run these in a WSL terminal (not PowerShell), in the project folder:

```bash
docker/run.sh build                 # once (10-30 min the first time; later only the code is rebuilt)
docker/run.sh check --no-llm        # is everything installed?
GPU=1 docker/run.sh llm-start       # Ollama in its own container, on the NVIDIA GPU (without GPU=1: CPU only)
docker/run.sh llm-pull              # the model qwen2.5:3b-instruct (about 2 GB, once)

# terminal 1: Gazebo + Nav2 + the robot with the forklift
docker/run.sh run ros2 launch actr_lab1 sim.launch.py

# terminal 2: check that everything runs, then the agent with the real LLM ...
docker/run.sh check --runtime
docker/run.sh exec ros2 launch actr_lab1 agent.launch.py planner:=ollama shutdown_when_done:=true
# ... or with the Mock planner (a new simulation first: after one mission the cubes are already in room A)
docker/run.sh exec ros2 launch actr_lab1 agent.launch.py planner:=mock shutdown_when_done:=true

# terminal 3 (optional): the production rules of ACT-R live, while the agent runs
tools/actr_live.sh
```

Useful arguments: `mission:="..."` (no spaces around `:=`), `fallback:=false`, `ablate:=gate`,
`gui:=false`, `arm:=false`, `gripper:=virtual`, `show_tokens:=true`, `llm_model:=...`. All of them:
`docker/run.sh exec ros2 launch actr_lab1 agent.launch.py --show-args` (and `sim.launch.py --show-args`
for the simulation ones like `gui` and `arm`).
Every run writes its logs to `~/actr_lab1_logs/<time>/` (`events.jsonl` and `results.jsonl`).
I used ROS 2 Humble (Ubuntu 22.04 in the container, Gazebo Classic 11). The code also has a path for
Foxy (`LAB_ROS_DISTRO=foxy`), but I did not test it.

### Only Python (no ROS, Python 3.8 or newer, no extra packages)

```bash
cd src/actr_lab1
python3 -m unittest discover -s test            # 491 tests
python3 -m actr_lab1.demo                       # a normal mission with a simulated robot
python3 -m actr_lab1.demo --scenario forbidden  # the mission asks for the restricted zone: the gate refuses
python3 -m actr_lab1.demo --list                # all scenarios
```

The tests use a fake ROS and a fake LLM server, so they run anywhere. Some tests need PyYAML
(`pip install pyyaml`) and one needs matplotlib; without them those tests are skipped.

### Experiments (no Gazebo needed)

```bash
cd src/actr_lab1
python3 -m actr_lab1.experiments.ablation --episodes 100 --out ../../results/ablation
python3 -m actr_lab1.experiments.learning --out ../../results/learning
python3 -m actr_lab1.experiments.compare_planners --out ../../results/compare   # Mock vs real LLM (needs Ollama)
```

With Docker the comparison can run in the container (Ollama started, simulation not running):
`docker/run.sh run python3 -m actr_lab1.experiments.compare_planners` - the results go to
`results/compare/`.

## Results

**Important:** the ablation and learning numbers come from my pure-Python simulation, not from Gazebo.
The "LLM" in these experiments is the Mock planner, which makes a mistake on purpose in 50 % of its
first answers (it is not a real language model; after it gets the error message it answers correctly).
The simulation also adds blocked paths, lost localization and Nav2 failures on purpose.

### Ablation (100 episodes per condition, the same episodes for all)

| Condition | Success | Unsafe runs | What it shows |
|---|---|---|---|
| full | 98 % | 0 % | the whole agent |
| no_gate | 98 % | 27 % | success stays the same, but in 27 % of the runs the robot enters the restricted zone |
| no_recovery | 21 % | 0 % | recovery after errors matters most for success |
| no_validator | 98 % | 0 % | same success, but more re-plans (0.32 vs 0.11) |
| no_repair | 99 % | 0 % | same success |
| no_memory | 98 % | 0 % | same success |
| no_learning | 100 % | 0 % | 2 episodes of difference, that is noise |
| bare (all off) | 19 % | 22 % | the baseline "without the architecture" |

The 95 % intervals of `full` and `no_learning` overlap, so I do not claim that difference. In my small
world the validator, the plan repair and the memory hardly change the success rate; the validator and
the repair reduce re-plans and gate refusals, and the memory effect is very small.

### Utility learning

- If the first waypoint of room A is blocked, the agent with learning makes 0.07 useless retries per
  episode on average, without learning 2.00.
- With the gate off, learning does not stop the robot from entering the restricted zone (10 runs out
  of 10). So utility learning is not a replacement for safety.

### Mock planner vs real LLM

`compare_planners` runs 12 missions (10 normal ones in Russian and English, 2 safety ones) with both
planners and saves, for every LLM call, the input state, the answer, the time and the tokens
(`decisions_*.jsonl`). It uses the simulated robot, not Gazebo. My results with qwen2.5:3b-instruct
are in `results/compare/summary.md` after I run it. The folder `examples/standin_output/` is only an
example of the format (there a test server answers, not a language model), so those numbers do not count.

## What I checked

Automatic tests: 491, they pass on Python 3.8, 3.10, 3.11 and 3.12. They use a fake ROS and a fake LLM
server, so they check my logic, not Gazebo.

On my laptop (Windows 11 + WSL2 + Docker Desktop, RTX 4070 laptop GPU). The first five checks were done
with the virtual gripper, before I added the forklift:

- [x] The Docker image builds and `docker/run.sh check` passes
- [x] Gazebo world, TurtleBot3, Gazebo services, Nav2 (localization and navigation)
- [x] Mission with the Mock planner in Gazebo: 8 steps, success
- [x] Safety test in Gazebo: the mission asks for the restricted zone, the gate denies it
      (`forbidden_zone`), the step is skipped, the robot never enters
- [x] Ollama on the GPU, and everything works without internet
- [ ] The forklift in Gazebo (pick and place with the pallets). New: so far tested only with the fake ROS
- [ ] Mock vs real LLM comparison (at least 10 runs)

## Limitations

1. My ACT-R is simplified: no spreading activation and no exact ACT-R timing.
2. The buffer access rights protect inside one program; they are not an OS sandbox.
3. The forklift uses the true pose from Gazebo for the docking (see above). With the forklift the lidar
   skips everything closer than 25 cm, so something that comes very close to the robot is not seen;
   a real robot would cut out only the box of its own load (for example with `laser_filters`). While
   the forklift works, the ACT-R cycle (and its emergency stop) waits for it (see Safety gate).
4. One robot (TurtleBot3 Burger) and one fixed world.
5. The mission text is read with simple patterns for typical Russian and English phrases (for the goal
   check); the plan itself comes from the LLM.
6. The ablation and learning numbers come from the simulation (see Results).

## Seminar 2, homework 5: what was asked and where it is

| Requirement | Where |
|---|---|
| A planner with the same interface `propose(buffers) -> Chunk` | `core/planners.py`: `OllamaPlanner` (= `RealLLMPlanner`), `OpenAICompatPlanner`, `MockLLMPlanner` |
| Structured output | JSON schema (`PLAN_SCHEMA`) in the request to Ollama, `extract_json`, the validator |
| A list of allowed tools | `navigate`, `pick`, `place`: `core/plan.py`, `core/gate.py` |
| A check before execution | `PlanValidator` (the plan) and `ActionGate` (every action) |
| A log of the input state and the chosen action | `events.jsonl`, `decisions_*.jsonl` |
| At least 10 runs with Mock and with the real LLM | `experiments/compare_planners.py` |
| No API keys in the code or in Git | the LLM is local; if a server needs a key, it is read from the variable `LLM_API_KEY`; `.gitignore` blocks `.env`, `*.key`, `*.pem` |

## Project structure

```
docker/            Dockerfile and run.sh (build and run)
tools/             world_layout.py, generate_world.py (Gazebo world + Nav2 map), preflight.sh (checks),
                   actr_live.sh (ACT-R rules live)
docs/              pictures of the world and of the demo scenarios
results/           ablation and learning results (simulation)
examples/          example output of compare_planners (test server)
src/actr_lab1/     the ROS 2 package
  actr_lab1/core/          ACT-R, gate, validator, planners, agent, SimRobot (no ROS)
  actr_lab1/ros/           the agent node, RosRobot (Nav2, TF, Gazebo, the forklift), arm_model.py
  actr_lab1/experiments/   ablation, learning, Mock vs LLM
  launch/ config/ worlds/ maps/ test/
```

The agent talks to the robot only through the `RobotAPI` interface (`core/robot_api.py`). There are two
implementations: `SimRobot` (pure Python) and `RosRobot` (ROS 2). That is why the agent code did not
change when I moved from my simple simulation to Gazebo. For the forklift the agent only asks the robot
how far from a cube to stop (`approach_radii`, `approach_margin`) and how far in front of it to put a
cube down (`drop_reach`); `SimRobot` does not have these, so the simulation results stayed exactly the
same. The agent node also makes "next to the cube" bigger when the robot has the forklift (1.3 m, and
1.5 m for the gate).

## Next: Lab 2

Lab 2 asks to replace the standard Nav2 controller with an LQG or PMP controller. The plan is a new
robot class with the same `RobotAPI` (for example a subclass of `RosRobot`) that takes the path from
Nav2 (`ComputePathToPose`) and follows it with my own controller through `/cmd_vel`. ACT-R, the LLM,
the validator and the gate stay as they are.
