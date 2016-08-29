
%include dependent library
addpath('./include/utils/');
addpath('./include/yael_matlab');

img_dir = './oxbuild_images/';

oxbuild_data   = './public_data/';

%load data 
nsift_fname    = [oxbuild_data 'oxford_nsift.uint32'];
sift_geo_fname = [oxbuild_data 'oxford_geom_sift.float'];
gnd_fname      = [oxbuild_data 'gnd_oxford.mat'];
codebook_path  = [oxbuild_data 'clust_preprocessed/oxford_codebook.fvecs'];
vw_path        = [oxbuild_data 'clust_preprocessed/oxford_vw.int32'];

%load  groud truth
load(gnd_fname);

%load sift geo data
Xgeom = load_ext(sift_geo_fname, 5); %5*n_feat

%compute sift index of images;
ndes  = load_ext(nsift_fname);  
cndes = [0 cumsum(double(ndes))];  %1*n_image

%load clustering data
codebook = load_ext(codebook_path , 128);  %128*n_center
vw_data  = load_ext(vw_path);  %1*n_feat
codebook_size = size(codebook , 2);

%load tmp data
load('tmp_data/idf_value.mat');
load('tmp_data/bow_result.mat','map','aps','result_flag','result_list');
load('tmp_data/ransac_data_20.mat','all_match_ransac','all_opt_matrix');

rerank_size = 200;
query_size  = size(qidx,1);
image_size  = length(imlist);
all_rerank_idx = zeros(rerank_size,query_size);

%set LBD parameters        
threshold_s = 5;     % for geometric filtering
threshold_num = 50;  % for one-to-one matching
threshold_tf = 3;    % for down weight 

tic
for k1 = 11:query_size 
    
    img1 = imread([ img_dir imlist{qidx(k1)} '.jpg']);
    feat1 = Xgeom(:,cndes(qidx(k1))+1 : cndes(qidx(k1)+1));
    vw1   = vw_data(:,cndes(qidx(k1))+1 : cndes(qidx(k1)+1));
    rerank_score = zeros(rerank_size,1);
    
    for k2 = 1:rerank_size 
        
        img2 = imread([ img_dir imlist{result_list(k2+1,k1)} '.jpg']);
        feat2_idx = result_list(k2+1,k1);
        feat2 = Xgeom(:,cndes(feat2_idx)+1 : cndes(feat2_idx+1));
        vw2   = vw_data(:,cndes(feat2_idx)+1 : cndes(feat2_idx+1));
                                
        match_point_ransac = all_match_ransac{k2,k1};
        opt_aff_matrix     = all_opt_matrix{k2,k1};

        %if we wang to display the match region of two images, just
        %uncomment following line.
        disp_match_features_vgg(img1,img2,feat1,feat2,match_point_ransac)
        
        % LBD step1: geometric filtering
        [match_point_ransac, ~] = Filter_area(feat1,feat2,match_point_ransac,opt_aff_matrix,threshold_s);                                              
                
        match_num = size(match_point_ransac,2);
        if (match_num >0)                         
            [unq_vw,~] = unique(vw1(match_point_ransac(1,:)));

            if(match_num < threshold_num)
                
                %LBD step2: one-to-one matching
                match_point_ransac = OneV1Match(match_point_ransac, vw1, idf_value);
                                
                %LBD step3: down weight
                local_vw_size = zeros(2,codebook_size);
                for k3 = 1: size(match_point_ransac,2)
                    idx_vw1 = vw1(match_point_ransac(1,k3));
                    local_vw_size(1,idx_vw1) =  local_vw_size(1,idx_vw1) +1;
                    idx_vw2 = vw2(match_point_ransac(2,k3));
                    local_vw_size(2,idx_vw2) =  local_vw_size(2,idx_vw2) +1;
                end
                local_tf = local_vw_size(1, vw1(match_point_ransac(1,:)));
                local_idf =  idf_value(vw1(match_point_ransac(1,:))) ;
                scoremap =  1 ./ ( 1*sqrt(local_tf) + 0);  
                scoremap(local_tf < threshold_tf)  = 1;                
                tmp_score = sqrt(local_idf)' .* scoremap;
                rerank_score(k2) = sum( tmp_score );                                
            else                
                tmp_score = sqrt(idf_value(vw1(match_point_ransac(1,:))));
                rerank_score(k2) = sum(tmp_score);                
            end            
        end        
        
        %disp(k2);
    end
    
    [~,rerank_idx]   = sort(rerank_score,'descend');
    all_rerank_idx(:,k1) = rerank_idx;
    tmp_result      = result_list(2:rerank_size+1,k1);
    result_list(2:rerank_size+1,k1) = tmp_result(rerank_idx);
    disp(k1);
end
toc

[map_rerank, aps_rerank,rerank_flag] = compute_map_select (result_list, gnd);

fprintf('\nmAP after BoW : %0.4f\nmAP after RANSAC&LBD : %0.4f\n',map, map_rerank);



%Our result:  

%FSM+GeoFitering:
%           rerank = 1000, mAP_rerank = 0.6509
%           rerank = 200,  mAP_rerank = 0.6307      

% FSM+1v1(threshold_num = 50):      
%           rerank = 1000, mAP_rerank = 0.6769, time = 70s (i5 CPU)
%           rerank = 200,  mAP_rerank = 0.6451, time = 32s  
 

% FSM+LBD(threshold_num = 50):      
%           rerank = 1000, mAP_rerank = 0.6811, time = 75s
%           rerank = 200,  mAP_rerank = 0.6488, time = 34s  
