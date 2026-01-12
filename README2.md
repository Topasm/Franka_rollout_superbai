# Server
### Docker Build
```bash
docker build -t openvla-server .
```

### Run OpenVLA Server
```bash
docker run -it --gpus all --ipc host --network host --rm -v /home/superb/workspace/.cache/huggingface/:/root/.cache/huggingface/ openvla-server
```

# Client
```bash
conda create -n client python=3.10 -y
conda activate client
pip install -r requirements_client.txt
conda install -c conda-forge fmt -y
```

### Install pylibfranka in client environment
```bash
conda activate client
cd ~/Desktop/libfranka_driver/libfranka/build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release \
      -DGENERATE_PYLIBFRANKA=ON \
      -DPython3_EXECUTABLE=/home/superb/anaconda3/envs/client/bin/python \
      -DCMAKE_PREFIX_PATH=/home/superb/anaconda3/envs/client \
      ..
cmake --build . -- -j$(nproc)

# Copy built files to client environment
sudo mkdir -p /home/superb/anaconda3/envs/client/lib/python3.10/site-packages/pylibfranka
sudo cp ~/Desktop/libfranka_driver/libfranka/build/libfranka.so* /home/superb/anaconda3/envs/client/lib/
sudo cp ~/Desktop/libfranka_driver/libfranka/build/pylibfranka/_pylibfranka.cpython-310-aarch64-linux-gnu.so /home/superb/anaconda3/envs/client/lib/python3.10/site-packages/pylibfranka/
sudo cp ~/Desktop/libfranka_driver/libfranka/pylibfranka/__init__.py /home/superb/anaconda3/envs/client/lib/python3.10/site-packages/pylibfranka/
echo '__version__ = "0.18.0"' | sudo tee /home/superb/anaconda3/envs/client/lib/python3.10/site-packages/pylibfranka/_version.py

# Verify installation
export LD_LIBRARY_PATH=/home/superb/anaconda3/envs/client/lib:$LD_LIBRARY_PATH
sudo -E /home/superb/anaconda3/envs/client/bin/python -c "import pylibfranka; print('Success:', pylibfranka.__version__)"
```

### Run Client (test connection)
```bash
export LD_LIBRARY_PATH=/home/superb/anaconda3/envs/client/lib:$LD_LIBRARY_PATH
sudo -E /home/superb/anaconda3/envs/client/bin/python vla-scripts/client.py
```

### Run Client Real (robot deploy)
```bash
export LD_LIBRARY_PATH=/home/superb/anaconda3/envs/client/lib:$LD_LIBRARY_PATH
sudo -E /home/superb/anaconda3/envs/client/bin/python vla-scripts/client_real.py
```

