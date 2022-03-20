#Train baseline 2x2 model fractional sigmoid payment & softmax allocation
python train.py --batch-size 20000 --num-examples 600000 \
--name 2x2_softmax_in_scratch --p_activation frac_sigmoid --a_activation softmax \
--num-epochs 200 --rho-incr-amount 0 --n-agents 2 --n-items 2

#Train baseline 2x2 model fractional linear sigmoid payment & sparsemax allocation
python train.py --batch-size 20000 --num-examples 600000 \
--name 2x2_sparsemax_in_linsigpmt_scratch --p_activation frac_sigmoid_linear --a_activation sparsemax \
--num-epochs 200 --rho-incr-amount 0 --n-agents 2 --n-items 2

#Train baseline 2x2 model fractional linear sigmoid payment & sparsemax allocation with certificate regularizer
python train.py --batch-size 20000 --num-examples 600000 \
--name 2x2_sparsemax_in_linsigpmt_scratch_fast --p_activation frac_sigmoid_linear --a_activation sparsemax \
--num-epochs 200 --rho-incr-amount 0 --n-agents 2 --n-items 2 --rs_loss

#Distill 2x2 model linear payment & sparsemax allocation
python train.py --batch-size 20000 --num-examples 600000 \
--name 2x2_sparsemax_linearpmt_distill --p_activation full_linear --a_activation sparsemax \
--num-epochs 200 --rho-incr-amount 0 --n-agents 2 --n-items 2 --teacher model/2x2_softmax_in_scratch.pt

#Distill 2x2 model linear payment & sparsemax allocation with certificate regularizer
python train.py --batch-size 20000 --num-examples 600000 \
--name 2x2_sparsemax_linearpmt_distill_fast --p_activation full_linear --a_activation sparsemax \
--num-epochs 200 --rho-incr-amount 0 --n-agents 2 --n-items 2 --teacher model/2x2_softmax_in_scratch.pt\
--rs_loss

#Sample testing scripts
python test.py --p_activation frac_sigmoid --a_activation softmax \
--model model/2x2_softmax_in_scratch.pt --n-agents 2

python test.py --p_activation frac_sigmoid_linear --a_activation sparsemax \
--model model/1x2_sparsemax_in_linsigpmt_scratch.pt --n-agents 1

python test.py --p_activation frac_sigmoid_linear --a_activation sparsemax \
--model model/2x2_sparsemax_in_linsigpmt_scratch_fast.pt --n-agents 2

python test.py --p_activation full_relu_clipped --a_activation sparsemax \
--model model/2x2_sparsemax_linearpmt_distill_fast.pt --n-agents 2


#Train baseline 1x2 model fractional sigmoid payment & softmax allocation
python train.py --batch-size 20000 --num-examples 600000 \
--name 1x2_softmax_in_scratch --p_activation frac_sigmoid --a_activation softmax \
--num-epochs 200 --rho-incr-amount 0 --n-agents 1 --n-items 2

#Train baseline 1x2 model fractional linear sigmoid payment & sparsemax allocation
python train.py --batch-size 20000 --num-examples 600000 \
--name 1x2_sparsemax_in_linsigpmt_scratch --p_activation frac_sigmoid_linear --a_activation sparsemax \
--num-epochs 200 --rho-incr-amount 0 --n-agents 1 --n-items 2

#Train baseline 1x2 model fractional linear sigmoid payment & sparsemax allocation with certificate regularizer
python train.py --batch-size 20000 --num-examples 600000 \
--name 1x2_sparsemax_in_linsigpmt_scratch_fast --p_activation frac_sigmoid_linear --a_activation sparsemax \
--num-epochs 200 --rho-incr-amount 0 --n-agents 2 --n-items 2 --rs_loss

#Distill 1x2 model linear payment & sparsemax allocation
python train.py --batch-size 20000 --num-examples 600000 \
--name 1x2_sparsemax_linearpmt_distill --p_activation full_linear --a_activation sparsemax \
--num-epochs 200 --rho-incr-amount 0 --n-agents 1 --n-items 2 --teacher model/1x2_softmax_in_scratch.pt \

#Distill 2x2 model linear payment & sparsemax allocation with certificate regularizer
python train.py --batch-size 20000 --num-examples 600000 \
--name 1x2_sparsemax_linearpmt_distill_fast --p_activation full_linear --a_activation sparsemax \
--num-epochs 200 --rho-incr-amount 0 --n-agents 1 --n-items 2 --teacher model/1x2_softmax_in_scratch.pt \
--rs_loss




