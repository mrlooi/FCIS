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


### Running a demo (example: sorting)
1. Download the pretrained model for sorting from our server (ask the vision team for the server password).
    The pretrained model for sorting is named "dora-box-envelope-0008.params" (Unless you know what you are doing, do not rename it)
    ```
    scp -r drml@10.0.9.33:~/Downloads/FCIS/model/dora-box-envelope-0008.params model/
    ```
2. Run the demo! 
    ```
    python ./fcis/demo.py --cfg ./config/dora_box_envelope.yaml --model ./model/dora-box-envelope-0008.params --img_dir ./data
    ```

### Using with ROS
The relevant ROS files are in the 'ros' folder. Place the 'msg', 'srv' and 'include' files in your catkin project. Example Ros C++ code for retrieving the data can be found in the 'ros/test' folder. 

