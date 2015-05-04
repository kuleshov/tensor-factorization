function [V1 Lambda misc] = no_tenfact(T, L, k)

% NO_TENFACT  Computes the CP decomposition of a tensor
%   [V, Lambda, flops, V0] = tendecomp(T, L, k) computes
%   the CP decomposition of a rank-k tensor via simultaneous
%   matrix diagonalization.
%
%   INPUTS:
%       T:      Third-order tensor
%       k:      Rank of T
%       L:      Number of random projections of T
%
%   OUTPUTS:
%       V1:     (d x k) matrix of tensor components of T
%       Lambda: (k x 1) vector of component weights
%       misc:   Structure with extra (diagnostic) output:
%         misc.flops:   Estimate of the number of flops performed
%         misc.V0:      matrix components obtained only form random projection
%
%   The algorithm first projects T onto L random vectors to obtain·
%   L projeted matrices, which are decomposed used the QRJ1D algorithm·
%   for joint non-orthogonal matrix diagonalization.
%   The resulting components are then used as plug-in estimates for the·
%   true components in a second projection step along V0. The joint decomposit
%   of this second set of matrices produces the final result V1.
%
%   Our implementation requires the MATLAB Tensor Toolbox v. 2.5 or greater.
%   The input tensor object must be constructed using the Tensor Toolbox.
%
%   For more information on the method see the following papers:
%
%   V. Kuleshov, A. Chaganty, P. Liang, Tensor Factorization via Matrix
%   Factorization, AISTATS 2015.
%
%   V. Kuleshov, A. Chaganty, P. Liang, Simultaneous diagonalization:
%   the asymmetric, low-rank, and noisy settings. ArXiv Technical Report.

p = size(T,1);
sweeps = [0 0];

% STAGE 1: Random projections

M = zeros(p, p*L);
W = zeros(p,L);

for l=1:L
    W(:,l) = randn(p,1);
    W(:,l) = W(:,l) ./ norm(W(:,l));
    M(:,(l-1)*p+1:l*p) = double(ttm(T,{eye(p), eye(p), W(:,l)'}));
end

[D, U, S] = qrj1d(M);

% calculate the true eigenvalues across all matrices
Ui = inv(U);
Ui_norms = sqrt(sum(Ui.^2,1));
Ui_normalized = bsxfun(@times, 1./Ui_norms, Ui);

dot_products = Ui_normalized'*W;
Lambdas = zeros(p,L);
for l=1:L
    Lambdas(:,l) = (diag(D(:,(l-1)*p+1:l*p)) ./ dot_products(:,l)) .* (Ui_norms.^2)';
end

% calculate the best eigenvalues and eigenvectors
[~, idx0] = sort(mean(abs(Lambdas),2),1,'descend');
Lambda0 = mean(Lambdas(idx0(1:k),:),2);
V = Ui_normalized(:, idx0(1:k));

% store number of sweeps
sweeps(1) = S.iterations;
sweeps(2) = S.iterations;

% STAGE 2: Plugin projections

W=Ui_normalized;
M = zeros(p, p*size(W,2));

for l=1:size(W,2)
    w = W(:,l);
    w = w ./ norm(w);

    M(:,(l-1)*p+1:l*p) = double(ttm(T,{eye(p), eye(p), w'}));
end

[D, U, S] = qrj1d(M);
Ui = inv(U);
Ui_norm=bsxfun(@times,1./sqrt(sum(Ui.^2)),Ui);
V1 = Ui_norm;
sweeps(2) = sweeps(2) + S.iterations;

Lambda = zeros(p,1);
for l=1:p
    Z = inv(V1);
    X = Z * M(:,(l-1)*p+1:l*p) * Z';
    Lambda = Lambda + abs(diag(X));
end
[~, idx] = sort(abs(Lambda), 'descend');
V1 = Ui_norm(:, idx(1:k));

misc = struct;
misc.V0 = V;
misc.sweeps = sweeps;

end
