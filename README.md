# UAV-AdversarialLearning
## Multiple GPU Training
```{r, engine='bash', count_lines}
#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python trainval_net_monitor.py --cuda --mGPUs --retraining_steps 250 --monitor_discriminator True --use_adversarial_loss True --use_utility_loss True --use_restarting --bs 8 --gamma 0.01 --depth 3 --save_iters 100 --n_minibatches 2
```
## Single GPU Testing
```{r, engine='bash', count_lines}
#!/bin/bash
for ((i=0; i<=10; i++))
do
        epoch=$(($i*1000/11914+1))
        ckpt=$((i*1000%11914))
        echo "$epoch"
        echo "$ckpt"
        CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda --checkepoch "$epoch" --checkpoint "$ckpt" --gamma 0.01
done

```

## Project Directory Layout
```
.
├── cfgs
├── data              # UAVDT dataset with annotation
├── images
├── lib
├── logs              # TensorBoard event files
├── models            # Trained model (w/ adversarial loss and w/o adversarial loss)
├── output
├── summaries         # Summary files recording the training and validation performance
├── README.md
├── _init_paths.py
├── bash_run.sh       # Run the testing in batch
├── demo.py
├── requirements.txt
├── test_net.py
├── trainval_net.py
└── trainval_net_monitor.py
```
