function   vwtest = quantizationVW_MA(vtest,codebook,vtrain_mean,knn)


%quantization 
nsift = size(vtest,2);
vwtest = zeros(knn,nsift);

slice = 1000;
lastid = 0;
for k1 = 1:nsift
    
    curr_vtest = desc_postprocess_iccv2013(vtest(:,lastid+(1:slice)), vtrain_mean);
    [vwtest(:, lastid + (1:slice)), ~] = yael_nn(codebook,curr_vtest,knn);
    
    lastid = lastid + slice;
    if(lastid / 10000 == round(lastid/10000))
        disp(lastid);
    end
    
    if(lastid+slice > nsift)
        slice = nsift - lastid;
        curr_vtest = desc_postprocess_iccv2013(vtest(:,lastid+(1:slice)), vtrain_mean);        
        [vwtest(:, lastid + (1:slice)), ~] = yael_nn(codebook,curr_vtest,knn);
        break;
    end
    
end