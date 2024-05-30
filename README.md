# Joystick Control User Interface
Starts up a Graphical User Interface that allows streaming of vessel commands for human control. 
Currently hardcoded to work for TitoNeri actuation structure

<p align="center" width="100%">
    <img width="33%" src="https://github.com/RAS-Delft/ras_joystick_control/assets/5917472/2161ca47-cb0b-45f9-b8db-09980b44ceff">
</p>

| Inputs | Outputs |
|--------|----------|
| Logitech Extreme 3D PRO (usb)| ros2 [Jointstate](https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/JointState.html) reference on /<vessel_id>/reference/actuation_prio |


## Use
Clone this repository in ros2_ws/src. Build ros2 workspace. Source workspace. Start with:
```shell
ros2 run joystick_control_ras joystickgui
```

Default joystick number is 0 if you only have 1 plugged in.
