import argparse
from data_loader import load_data
from train import train
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def print_setting(args):
    def logger(str):
        now = time.strftime("[%Y-%m-%d %H:%M:%S] ", time.localtime())  
        print(now+str)
        with open("../data/"+args.dataset+"/test_output.log",'a') as f:
            f.write(now+str+"\n")


    assert args.use_context or args.use_path
    logger("")
    logger('=============================================')
    logger("method: original")
    logger('dataset: ' + args.dataset)
    logger('epoch: ' + str(args.epoch))
    logger('batch_size: ' + str(args.batch_size))
    logger('dim: ' + str(args.dim))
    logger('l2: ' + str(args.l2))
    logger('lr: ' + str(args.lr))
    logger('feature_type: ' + args.feature_type)

    logger('use relational context: ' + str(args.use_context))
    if args.use_context:
        logger('context_hops: ' + str(args.context_hops))
        logger('neighbor_samples: ' + str(args.neighbor_samples))
        logger('neighbor_agg: ' + args.neighbor_agg)

    logger('use relational path: ' + str(args.use_path))
    if args.use_path:
        logger('max_path_len: ' + str(args.max_path_len))
        logger('path_type: ' + args.path_type)
        if args.path_type == 'rnn':
            logger('path_samples: ' + str(args.path_samples))
            logger('path_agg: ' + args.path_agg)
    logger('=============================================')
    logger("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, help='use gpu', action='store_true')

    '''
    # ===== FB15k ===== #
    parser.add_argument('--dataset', type=str, default='FB15k', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

    # settings for relational context
    parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=32, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # settings for relational path
    parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    parser.add_argument('--max_path_len', type=int, default=2, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    '''

    '''
    # ===== FB15k-237 ===== #
    parser.add_argument('--dataset', type=str, default='FB15k-237', help='dataset name')
    parser.add_argument('--epoch', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

    # settings for relational context
    parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=32, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # settings for relational path
    parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    '''    

    # '''
    # ===== wn18 ===== #
    parser.add_argument('--dataset', type=str, default='wn18', help='dataset name')
    parser.add_argument('--epoch', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

    # settings for relational context
    parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=16, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # settings for relational path
    parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    # '''

    '''
    # ===== wn18rr ===== #
    parser.add_argument('--dataset', type=str, default='wn18rr', help='dataset name') #==!
    # parser.add_argument('--dataset', type=str, default='testData', help='dataset name')

    parser.add_argument('--epoch', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    # parser.add_argument('--batch_size', type=int, default=4, help='batch size')

    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

    # settings for relational context
    parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # settings for relational path
    parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    parser.add_argument('--max_path_len', type=int, default=4, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    '''

    '''
    # ===== NELL995 ===== #
    parser.add_argument('--dataset', type=str, default='NELL995', help='dataset name')
    parser.add_argument('--epoch', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

    # settings for relational context
    parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')
 
    # settings for relational path
    parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    '''

    '''
    # ===== DDB14 ===== #
    parser.add_argument('--dataset', type=str, default='DDB14', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

    # settings for relational context
    parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='cross', help='neighbor aggregator: mean, concat, cross')

    # settings for relational path
    parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    parser.add_argument('--max_path_len', type=int, default=4, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    '''

    args = parser.parse_args()
    print_setting(args)
    data = load_data(args)
    train(args, data)

## 算邻居的时候当做无向图考虑的，算path的时候也当做无向图考虑
## ==! 表示有注释 ?表示有疑问
if __name__ == '__main__':
    main()
