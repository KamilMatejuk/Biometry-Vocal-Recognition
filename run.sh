function run() {
    name="full_ds"
    echo "Running $name"
    if [ ! -f "data/checkpoints/GhostFace/$name.chpt" ]; then
        echo "Copy pretrained data/checkpoints/GhostFace/pretrained_ghostnet_v2_1.0.chpt"
        cp data/checkpoints/GhostFace/pretrained_ghostnet_v2_1.0.chpt data/checkpoints/GhostFace/$name.chpt
    fi
    echo "ghost:" > config.temp.yml
    echo "  $name:" >> config.temp.yml
    echo "    num_classes: 8269" >> config.temp.yml
    echo "    width: 1.0" >> config.temp.yml
    echo "    dropout: 0.5" >> config.temp.yml
    echo "    momentum: 0.9" >> config.temp.yml
    echo "    lr: 1e-3" >> config.temp.yml
    echo "    wd: 0.0" >> config.temp.yml
    echo "    bs: 16" >> config.temp.yml
    python main.py -m ghost -a train --name $name --config config.temp.yml
}



run
# run
# run
# run
# run
# run