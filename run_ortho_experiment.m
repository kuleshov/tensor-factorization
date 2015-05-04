function [tpm_err, diag_err, nojd_err] = run_ortho_experiment(p, k, epsilon, tries, outfile)
    
addpath('/path/to/tensor_toolbox_2.5');
addpath('/path/to/tensorlab');
addpath('./bin')

randn('seed',0)

% generate tensors

TCell = cell(tries,1);
VCell = cell(tries,1);

for t=1:tries
    [Vsvd, Ssvd, Usvd] = svd(randn(p,p));
    lambda = randn(k,1).^2;
    V_true = Vsvd(:,1:k);

    T_true = ktensor(lambda, {V_true, V_true, V_true});
    [V_noise1, ~, ~] = svd(randn(p,p));
    lambda_noise1 = randn(p,1).^2;
    T_noise = ktensor(lambda_noise1, {V_noise1, V_noise1, V_noise1});

    T = (T_true) + epsilon*(T_noise);
    
    TCell{t, 1} = T;
    VCell{t, 1} = V_true;
end

out = fopen(outfile, 'w');
fclose(out);


% tensor power method

tpm_err = 0.0;
flops_tpm = 0;

for t=1:tries
    [V_tpm, ~, N] = tpm(TCell{t,1}, 1, p, k);
    V_true = VCell{t,1};
    flops_tpm = flops_tpm + p^3 * N;
    sum_err = 0.0;
    for x=1:size(V_true,2)
        V_1 = repmat(V_true(:,x), 1, size(V_tpm,2));
        sum_err = sum_err + min(min([sqrt(sum((V_tpm - V_1).^2,1)); sqrt(sum((V_tpm + V_1).^2,1))]));
    end
    tpm_err = tpm_err + sum_err/size(V_true,2);
%     sum_err/size(V_true,2)
end

tpm_err = tpm_err / tries;
flops_tpm = flops_tpm / tries;

fprintf('tpm\t%f\t%f\n', tpm_err, flops_tpm);

out = fopen(outfile, 'a');
fprintf(out, 'tpm\t%f\t%f\n', tpm_err, flops_tpm);
fclose(out);


% joint diagonalization

diag_err = 0.0;
diag_err0 = 0.0;
flops_ojd = [0 0];

for t=1:tries
    
    if p >= 200
        L = 20;
    elseif p >= 100 && p < 200
        L = 10;
    else
        L = 5;
    end

    [V_tjd, ~, misc] = tenfact(TCell{t,1}, L, k);
    flops = misc.flops;
    V_tjd0 = misc.V0;
    V_true = VCell{t,1};
    flops_ojd = flops_ojd + flops;
    sum_err = 0.0;
    for x=1:size(V_true,2)
        V_1 = repmat(V_true(:,x), 1, size(V_tjd,2));
        sum_err = sum_err + min(min([sqrt(sum((V_tjd - V_1).^2,1)); sqrt(sum((V_tjd + V_1).^2,1))]));
    end
    diag_err = diag_err + sum_err/size(V_true,2);
%     sum_err/size(V_true,2)

    sum_err = 0.0;
    for x=1:size(V_true,2)
        V_1 = repmat(V_true(:,x), 1, size(V_tjd0,2));
        sum_err = sum_err + min(min([sqrt(sum((V_tjd0 - V_1).^2,1)); sqrt(sum((V_tjd0 + V_1).^2,1))]));
    end
    diag_err0 = diag_err0 + sum_err/size(V_true,2);
end

diag_err = diag_err / tries;
diag_err0 = diag_err0 / tries;
flops_ojd = flops_ojd / tries;

fprintf('ojd0\t%f\t%f\n', diag_err0, flops_ojd(1));
fprintf('ojd1\t%f\t%f\n', diag_err, flops_ojd(2));

out = fopen(outfile, 'a');
fprintf(out, 'ojd0\t%f\t%f\n', diag_err0, flops_ojd(1));
fprintf(out, 'ojd1\t%f\t%f\n', diag_err, flops_ojd(2));
fclose(out);

    
% non orthognal joint diagonalization

nojd_err = 0.0;
nojd_err1 = 0.0;
sweeps_nojd = [0 0];
    
for t=1:tries
    if p <= 150
        [V1, ~, misc] = no_tenfact(TCell{t,1}, p, k);
        V_nojd = misc.V0;
        sweeps = misc.sweeps;
        V_true = VCell{t,1};
        sweeps_nojd = sweeps_nojd + sweeps;
        sum_err = 0.0;
        for x=1:size(V_true,2)
            V_1 = repmat(V_true(:,x), 1, size(V_nojd,2));
            sum_err = sum_err + min(min([sqrt(sum((V_nojd - V_1).^2,1)); sqrt(sum((V_nojd + V_1).^2,1))]));
        end
        nojd_err = nojd_err + sum_err/size(V_true,2);

        sum_err = 0.0;
        for x=1:size(V_true,2)
            V_1 = repmat(V_true(:,x), 1, size(V1,2));
            sum_err = sum_err + min(min([sqrt(sum((V1 - V_1).^2,1)); sqrt(sum((V1 + V_1).^2,1))]));
        end
        nojd_err1 = nojd_err1 + sum_err/size(V_true,2);
    end
end

nojd_err = nojd_err / tries;
nojd_err1 = nojd_err1 / tries;
sweeps_nojd = sweeps_nojd / tries;

fprintf('nojd0\t%f\t%f\n', nojd_err, sweeps_nojd(1));
fprintf('nojd1\t%f\t%f\n', nojd_err1, sweeps_nojd(2));

out = fopen(outfile, 'a');
fprintf(out, 'nojd0\t%f\t%f\n', nojd_err, sweeps_nojd(1));
fprintf(out, 'nojd1\t%f\t%f\n', nojd_err1, sweeps_nojd(2));
fclose(out);


% alternating least squares

als_err = 0.0;
iters_als = 0;

for t=1:tries
    
    options.Algorithm = @cpd_als;
    [X out] = cpd(double(TCell{t,1}),min(k,p), options);
    V_true = VCell{t,1};
    iters_als = iters_als + out.Algorithm.iterations;
    V_als = bsxfun(@times,1./sqrt(sum(X{1,1}.^2)),X{1,1});
    sum_err = 0.0;
    for x=1:size(V_true,2)
        V_1 = repmat(V_true(:,x), 1, size(V_als,2));
        sum_err = sum_err + min(min([sqrt(sum((V_als - V_1).^2,1)); sqrt(sum((V_als + V_1).^2,1))]));
    end
    als_err = als_err + sum_err/size(V_true,2);
end

als_err = als_err / tries;
iters_als = iters_als / tries;

fprintf('als\t%f\t%f\n', als_err, iters_als);

out = fopen(outfile, 'a');
fprintf(out, 'als\t%f\t%f\n', als_err, iters_als);
fclose(out);

% lathauwer's method

lath_err = 0.0;
iters_lath = 0;
    
for t = 1:tries
    if p <= 100
        options.Algorithm = @cpd3_sd;
        [X out] = cpd(double(TCell{t,1}),min(k,p), options);
        V_true = VCell{t,1};
        iters_lath = iters_lath + out.Algorithm.SDOutput.iterations;
        V_lath = bsxfun(@times,1./sqrt(sum(X{1,1}.^2)),X{1,1});
        sum_err = 0.0;
        for x=1:size(V_true,2)
            V_1 = repmat(V_true(:,x), 1, size(V_lath,2));
            sum_err = sum_err + min(min([sqrt(sum((V_lath - V_1).^2,1)); sqrt(sum((V_lath + V_1).^2,1))]));
        end
        lath_err = lath_err + sum_err/size(V_true,2);
    end
end

lath_err = lath_err / tries;
iters_lath = iters_lath / tries;

fprintf('lath\t%f\t%f\n', lath_err, iters_lath);

out = fopen(outfile, 'a');
fprintf(out, 'lath\t%f\t%f\n', lath_err, iters_lath);
fclose(out);
end
