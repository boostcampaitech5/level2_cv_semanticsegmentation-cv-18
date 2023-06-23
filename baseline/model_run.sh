num=319
model_list='deeplabv3plus' # pspnet' #deeplabv3  
encoder_list='r101' #effb5 r151' # effb3 effb5 r151'
opt_list='Adam' # AdamW' # SGD'
loss_list='dice' # tversky softBCE' # BCE'
aug_list='clahe1' #clahe1 clahe2' # clahe centercrop2' #centercrop2' # base  center_empty' # center_grid gridmask cropnonempty grid_empty'
sch_list=None #'CosineAnnealingLR' # StepLR'

num_workers=4
batch_size=4
learning_rate='1e-4'
max_epoch='50'


for model in $model_list
do
    for encoder in $encoder_list
    do
        for opt in $opt_list
        do
            for loss in $loss_list
            do
                for aug in $aug_list
                do
                    for sch in $sch_list
                    do
                        for epoch in $max_epoch
                        do
                            echo "exp_num:$num , model:$model, encoder:$encoder, opt:$opt, loss:$loss, aug:$aug, lr_scheduler: $sch" #, mixed:True" #
                            exp_name="${num}_${model}_${encoder}_${opt}_${loss}_${aug}_${sch}_resize1024" #_mixed" #_${lr}" # 

                            python custom_train.py\
                            --exp_name $exp_name\
                            --model $model\
                            --encoder $encoder\
                            --optimizer $opt\
                            --loss $loss\
                            --aug $aug\
                            --num_workers $num_workers\
                            --batch_size $batch_size\
                            --learning_rate $learning_rate\
                            --max_epoch $epoch
                        
                            num=`expr $num + 1`
                        done
                    done
                done
            done 
        done    
    done
done

