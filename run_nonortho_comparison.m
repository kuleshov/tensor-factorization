outfile = 'non-ortho-results.txt'
p = 20;
k = 20;
epsilons = [0.001, 0.00075, 0.0005, 0.00025, 0.0001];
tries = 10

for e=1:size(epsilons,2)
    epsilon = epsilons(1,e)
    run_nonortho_experiment(p, k, epsilon, tries, 'stdout');
end
