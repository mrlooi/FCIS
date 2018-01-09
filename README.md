## NOTE: THIS IS A MODIFIED REPO OF https://github.com/msracver/FCIS


### Installation on Debian 9.0 (Dorabot version)
Note that this installation is entirely dependent on the current Dorabot Debian 9.0 setup environment (see http://pha.dorabot.com/w/setup_software_development_environment/) and purely customized for Dorabot usage. It is not guaranteed to work in a different environment setup.
Assumes CUDA (cuda 8) and CUDNN are installed in /usr, as in the environment installation script

1. Python prerequisites:
	```
	sudo apt-get install libopenblas-dev python-h5py
	sudo pip install Cython easydict hickle
	```
2. Softlink /usr/lib64 (if it doesn't exist already)
    ```
    sudo ln -s /usr/lib /usr/lib64;
    ```
3. Install mxnet fork (http://gitlab.dorabot.com/vincent/mxnet_debian/)
4. Download pre-compiled FCIS library .so files from our server (ask the vision team for the server password)
    ```
    scp -r drml@10.0.9.33:~/Downloads/FCIS/lib $(your_fcis_folder_root);
    ```
5. Build FCIS libs
    ```
	sh ./init.sh;
	```


### Running a demo
1. You will need 2 files to run the network: config file and model file. 
Download the pretrained model and config from our RW server (RW/Vision/FCIS/project, where 'project' is currently 'loading' or 'sorting')
2. Run the demo! 
    ```
    python ./fcis/demo.py --cfg <config_file> --model <model_file> --img_dir ./data
    ```

### Using with ROS
The relevant ROS files are in the 'ros' folder. Place the 'msg', 'srv' and 'include' files in your catkin project. Example Ros C++ code for retrieving the data can be found in the 'ros/test' folder. 
    ```
    python ./fcis/ros_demo.py --cfg <config_file> --model <model_file> [--publish]
    ```
