outfile = 'ortho-results.txt'
p = 50;
k = 50;
epsilons = [0.01, 0.0075, 0.005, 0.0025, 0.001];
% epsilons = [0.001, 0.0075, 0.005, 0.0025, 0.0001]
tries = 25;

for e=1:size(epsilons,2)
    epsilon = epsilons(1,e)
    run_ortho_experiment(p, k, epsilon, tries, outfile);
end
