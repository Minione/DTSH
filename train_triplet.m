function [net, U] = train_triplet (X1, L1, U, net, iter , lr, eta, margin, fid)
    N = size(X1,4);
    batchsize = 128;

    index = randperm(N);
    M1 = 50;
    M2 = 50;
    numCategory = 10;
    tripletPerImg = M1 * M2 * (numCategory - 1);
    for j = 0:ceil(N/batchsize)-1
        batch_time=tic;

        %% random select a minibatch and sample triplets
        ix = index((1+j*batchsize):min((j+1)*batchsize,N));
        S = tripletSampling (L1, ix, 1:N, M1, M2, numCategory);

        %% load and preprocess an image
        im = X1(:,:,:,ix);
        im_ = single(im); % note: 0-255 range
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
        im_ = im_ - repmat(net.meta.normalization.averageImage,1,1,1,size(im_,4));
        im_ = gpuArray(im_);

        %% run the CNN
        res = vl_simplenn(net, im_);
        U0 = squeeze(gather(res(end).x))';
        U(ix,:) = U0; 

        %% compute the loss and gradient
        T = zeros(numel(ix), tripletPerImg);
        curIdx = 0;
        for k = 1:numel(ix)
            T(k, :) = U(S(curIdx + 1, 1), :) * U(S(curIdx + 1:curIdx + tripletPerImg, 2), :)' / 2 -...
                      U(S(curIdx + 1, 1), :) * U(S(curIdx + 1:curIdx + tripletPerImg, 3), :)' / 2 -...
                      margin;
            curIdx = curIdx + tripletPerImg;
        end
        A = 1 ./ (1 + exp(-T));
        loss = - sum(log(A(:))) / tripletPerImg / numel(ix);
        qloss = U0 - sign(U0);
        qloss = qloss .* qloss;
        qloss = sum(qloss(:)) / numel(ix);

        A = (1 - A);

        curIdx = 0;
        dJdU = [];
        for k = 1:numel(ix)
            tmp = U(S(curIdx + 1:curIdx + tripletPerImg, 2), :) -...
                  U(S(curIdx + 1:curIdx + tripletPerImg, 3), :);
            dJdU(k, :) = A(k, :) * tmp;
            curIdx = curIdx + tripletPerImg;
        end
        dJdU = single(dJdU);

        dJdU = dJdU - 4 * eta * (U0 - sign(U0));

        dJdoutput = gpuArray(reshape(dJdU',[1,1,size(dJdU',1),size(dJdU',2)]));
        res = vl_simplenn( net, im_, dJdoutput);

        %% update the parameters of CNN
        net = update(net , res, lr, N);
        batch_time = toc(batch_time);

        fprintf(' iter %d  batch %d/%d (%.1f images/s) ,lr is %d likelihood loss: %f quantization loss: %f\n', iter, j+1,ceil(size(X1,4)/batchsize), batchsize/ batch_time,lr, loss, qloss);
        fprintf(fid, ' iter %d  batch %d/%d (%.1f images/s) ,lr is %d likelihood loss: %f quantization qloss: %f\n', iter, j+1,ceil(size(X1,4)/batchsize), batchsize/ batch_time,lr, loss, qloss);
    end
end
