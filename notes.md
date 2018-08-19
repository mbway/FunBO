There is a potential problem whereby the uncertainty is very large at the edges of the domain since the initial phase usually stays close to the center on average, so the first Bayesian trial is ruined by picking extreme values everywhere
- possible fix: introduce a prior to stay close to where you think the function should be, weight the acquisition function accordingly
- possible fix: make sure to test extreme values in the initial phase


may want to stay close to the previous extracted function, but this is local optimisation rather than global

the main consumer of time is the extraction, not the fitting, using BFGS is very expensive so a number of random samples can be used instead, however this still requires many queries

it is very expensive to perform lots of independent local optimisations for each control point. The effort to approximate the gradient and Hessian is repeated for each control point for example without any sharing of information. Leads to a lot of computation.


analytical gradients do seem to require slightly fewer evaluations, however because there are still a huge number of evaluations, the added cost from computing the gradient is far greater and so makes things worse.

it appears that querying the acquisition function at a single point has a lot of overhead compared to querying at many points in a single call.
    - This makes local optimisation algorithms less effective since they are sequential whereas choosing randomly can benefit from the overhead.
    - Although in the current implementation, there is still one query (of many points) per control point.
        - Technically all the queries could be carried out with a single query to make the optimisation of the control points from many 1D optimisations to a single ND optimisation.
        - Since each control point is independent, you can still take the best value for each control point over all the runs, rather than taking the best overall solution.
    - further testing seems to suggest that the gap grows further with more training data.
    - so long as you query in reasonably sized batches, eg >100 then the average time per sample isn't too bad, the problem is mainly when querying at very few points at a time <10
        - when querying the gradient, the time per point actually increases past a batch size of 100
    - there is even more overhead when querying the gradient one point at a time
        - f: batch of 10,000 is 15-20 times faster than batch of 1 (10,000 query points)
        - gradient: batch of 100 is 40 times faster than batch of 1 (10,000 query points)

querying the analytical gradient of the GP is approximately 3x slower than querying the GP normally, but this is probably better than approximating numerically.
    - To approximate the gradient at x requires an evaluation at x, then D more evaluations by perturbing each dimension of x, so D+1 evaluations where D is the number of dimensions.
    - in addition, the queries used to approximate the gradient are made individually, which has huge overhead (as discussed above)

Thoughts on elastic extraction
    - from Wikipedia: "For problems where finding an approximate global optimum is more important than finding a precise local optimum in a fixed amount of time, simulated annealing may be preferable to alternatives such as gradient descent."
        - the problem I am dealing with is indeed of this kind, so I will focus on simulated annealing rather than gradient descent
    - since optimising the output values at each control point is a very high dimensional problem I think it is justified to perturb a small subset of the dimensions at once, so that the solution improves 
        - since the reward for the function fit is independent at each control point, the 'valleys' of the objective function are axis-aligned, the 'handbook of global optimization' discusses the challenge with simulated annealing where anisotropic perturbations should be made if the direction to the global optimum is diagonal.
            - for my problem, picking a random subset of dimensions to perturb while leaving the others constant seems like a valid approach
        - with the addition of the elastic energy there is a little correlation, but it should be fine
        - to minimise the number of GP queries, could perturb every dimension and query the acquisition function at the new values, but then propose them in several updates of a few dimensions being changed at a time rather than all together.
        - Siarry et al., 1997 used this approach


