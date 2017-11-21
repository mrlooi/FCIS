## NOTE: THIS IS A MODIFIED REPO OF https://github.com/msracver/FCIS


### Installation on Debian
1. Python prerequisites:
	```
	sudo apt-get install libopenblas-dev
	sudo apt-get install python-h5py
	sudo pip install Cython
	sudo pip install easydict
	sudo pip install hickle
	```
2. Custom installation (for Dorabot -> must have cuda 8 installed)
	Assumes CUDA and CUDNN are installed in /usr, as per DD's environment installation script
	```
	sudo ln -s /usr/lib /usr/lib64;
	
	scp -r drml@10.0.9.33:~/Downloads/mxnet $(MXNET);
	scp -r drml@10.0.9.33:~/Downloads/FCIS/lib $(FCIS);
	
	cd $(FCIS);
	sh ./init.sh;

	cp -r $(FCIS)/fcis/operator_cxx/* $(MXNET)/src/operator/contrib/
	cd $(MXNET); 
	make -j $(nproc) USE_OPENCV=0 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/ USE_CUDNN=1;
	cd python;
	sudo python setup.py install;

	```
