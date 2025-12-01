we have nvidia's ModelOpt in this repo (see pixi), and we have yolo11 (see models/yolo11)

now we want to test modelopt with yolo11, that is, using ModelOpt to find the best mixed precision quantization scheme for yolo11 (fp16/int8).

find out how, and develop a plan in context/tasks/working/task-quantize-yolo11-by-modelopt.md, with
- what to do
- major milestones to do it, each milestone with a short description, and a section
- for each section, create a TODO list of tasks to do it, each to do item is like `- [ ] Job-<section-number>-<index>: task description`