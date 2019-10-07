# beam search
CUDA_VISIBLE_DEVICES=2 python main.py --test ./data/yelp/large/model/2_256_2019-05-10-03:12:37/9_expansion_model.tar --batch_size 128 --corpus yelp/large --dmax 5 --hidden 256 

# sample
CUDA_VISIBLE_DEVICES=2 python main.py --test ./data/yelp/large/model/2_256_2019-05-10-03:12:37/9_expansion_model.tar --batch_size 128 --corpus yelp/large --dmax 5 --hidden 256 --sample  
