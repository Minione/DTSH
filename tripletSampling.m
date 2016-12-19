% images in idx1 as query
% for each query, M1 postive
% for each positive, M2 negative for each category
function R = tripletSampling (label, idx1, idx2, M1, M2, numCategory)
	L1 = label(idx1);
  	L2 = label(idx2);
  	L1 = single(L1);
  	L2 = single(L2);

	R = zeros(M1 * M2 * (numCategory - 1) * numel(idx1), 3);

	categoryIdx = {};
  	flag = ones(numel(idx2), 1);
  	flag(idx1) = 0; % Here assumes idx2 is always 1:N
	for i = 1:numCategory
		categoryIdx{i} = find((L2 == i - 1) & flag);
	end

	tripletPerImg = M1 * M2;
	curIdx = 0;
	for i = 1:numel(idx1)
		positiveIdx = categoryIdx{L1(i) + 1};
		for j = 0:(numCategory - 1)
	  		if (j ~= L1(i))
				negativeIdx = categoryIdx{j + 1};
		  		tmp1 = randperm(numel(positiveIdx), M1);
		  		tmp2 = randperm(numel(negativeIdx), M2);

		  		[p, q] = meshgrid(idx2(positiveIdx(tmp1)), idx2(negativeIdx(tmp2)));

				R((curIdx + 1):(curIdx + tripletPerImg), 2:3) = [p(:) q(:)];
				R((curIdx + 1):(curIdx + tripletPerImg), 1) = idx1(i);

				curIdx = curIdx + tripletPerImg;
			end
		end
	end
end
