### Install pylibfranka in client environment
```bash
conda activate client
cd ~/Desktop/libfranka_driver/libfranka/build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release -DGENERATE_PYLIBFRANKA=ON -DPython3_EXECUTABLE=/home/superb/anaconda3/envs/client/bin/python ..
cmake --build . -- -j$(nproc)
sudo cmake --install .
```

### Run Client (test connection)
```bash
sudo /home/superb/anaconda3/envs/client/bin/python vla-scripts/client.py
```

### Run Client Real (robot deploy)
```bash
sudo /home/superb/anaconda3/envs/client/bin/python vla-scripts/client_real.py
```