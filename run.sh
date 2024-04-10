function run() {
    local dropout=$1
    local momentum=$2
    local lr=$3
    local bs=$4
    local name="hiperparams_d${dropout}_m${momentum}_lr${lr}_bs${bs}"

    echo "Running $name"
    # TODO fix path to pretrained checkpoint
    if [ ! -f "data/checkpoints/GhostFace/$name.chpt" ]; then
        echo "Copy pretrained data/checkpoints/GhostFace/pretrained_ghostnet_v2_1.0.chpt"
        cp data/checkpoints/GhostFace/pretrained_ghostnet_v2_1.0.chpt data/checkpoints/GhostFace/$name.chpt
    fi
    # TODO change model from ghost
    echo "ghost:" > config.temp.yml
    echo "  $name:" >> config.temp.yml
    echo "    num_classes: 100" >> config.temp.yml
    echo "    dropout: ${dropout}" >> config.temp.yml
    echo "    momentum: ${momentum}" >> config.temp.yml
    echo "    lr: ${lr}" >> config.temp.yml
    echo "    bs: ${bs}" >> config.temp.yml
    python main.py -m ghost -a train --name $name --config config.temp.yml
}

for drop in 0.0 0.25 0.50; do
    for mom in 0.0 0.5 0.9; do
        for lr in 1e-1 1e-2 1e-3 1e-4; do
            for bs in 16 32 64; do
                run $drop $mom $lr $bs
            done
        done
    done
done
