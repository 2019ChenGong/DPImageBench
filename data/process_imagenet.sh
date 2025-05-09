for i in `seq 0 29`
    do
        python process_dataset.py --worker_id $i --num_workers 30 --data_dir /src/dataset/imagenet/ImageNet/train --new_dir /src/dataset/imagenet/imagenet_32/train --image_size 64 &
    done