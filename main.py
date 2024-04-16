from src.utils import *
from src import train
from configs import *
use_cuda = False

output_dim_dict = {
    'mosi': 1,
    'mosei': 1,
}

criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}


torch.set_default_tensor_type('torch.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True
set_random_seed(args.seed)
train_loader, valid_loader, test_loader, num_train_optimization_steps = set_up_data_loader(args)
hyp_params = args
hyp_params.use_cuda = use_cuda
hyp_params.device = device
hyp_params.name = str.upper(args.name.strip())
hyp_params.output_dim = output_dim_dict.get(args.dataset, 1)
hyp_params.criterion = criterion_dict.get(args.dataset, 'L1Loss')
hyp_params.ACOUSTIC_DIM = ACOUSTIC_DIM
hyp_params.VISUAL_DIM = VISUAL_DIM
hyp_params.TEXT_DIM = TEXT_DIM

if __name__ == '__main__':
    train.train(hyp_params, train_loader, valid_loader, test_loader, num_train_optimization_steps)

