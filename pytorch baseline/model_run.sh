num=350
model_list='hrnetocr' #deeplabv3plus' #unet2plus' #hrnetocr' #unet2plus' #deeplabv3plus' #hrnetocr' #deeplabv3plus' #unet2plus' # pspnet' #deeplabv3  
encoder_list=None #r152' #r101' #effb5 ' # effb3'
opt_list='Adam' # AdamW' # SGD'
loss_list='dice' #dicefocal' # ' # tversky softBCE' # BCE'
aug_list='base2' # brightclaherotate' #brightclahe' # bright2' # base2' #clahe2 bright2' # clahe3 bright2' # brightclahe' # clahe4 bright rotate  clahe1 clahe2' # clahe centercrop2' #centercrop2' # base  center_empty' # center_grid gridmask cropnonempty grid_empty'
sch_list=None # 'CosineAnnealingLR' # StepLR'

num_workers=2
batch_size=4
learning_rate='1e-3'
max_epoch='25'


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
                            exp_name="${num}_${model}_${encoder}_${opt}_${loss}_${aug}_${learning_rate}_resized1024" #_${sch}_copypaste(k=9)" #_${sch}" #_mixed" #_${lr}" # 

                            python custom_train.py\
                            --exp_name $exp_name\
                            --k 9\
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

#                            --lr_scheduler $sch\
