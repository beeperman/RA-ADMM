%% f(x) = 1/2*x'*M*x, g(z) = 1/2*z'*N*z, minimize f(x) + g(z), subject to x=z
d = 20;

M = rand(d,d); M = 1*rand(1,1)*(M+M' + 20*eye(d));
N = rand(d,d); N = 0.001*rand(1,1)*(N+N' + 20*eye(d));
A = rand(d/2,d);
F_L = max(max(eig(M)),max(eig(N)));
F_m = min(min(eig(M)),min(eig(N)));
F_k = F_L/F_m
%%

rho = min(1/3*min([min(eig(M)), min(eig(N))]))

max_iter = 150;

x0 = 1*rand(d,1);
z0 = 1*rand(d,1);
v0 = 1*rand(d,1);
%%
m = 1 / (max(max(eig(M)), max(eig(N))) + rho);
L = 1 / (rho);
r=0.5;

kappa = L / m;
eta = 1 - 1/kappa^r



alpha = kappa * (1-eta)^2 * ( 1 + eta) * rho;
beta = kappa*eta^3/(kappa - 1);
gamma = eta^3 / ( (kappa-1) * (1-eta)^2 * (1+eta) );
lambda = m^2*(kappa-kappa*eta^2-1)/(2*eta*(1-eta));
mu = (1+eta)*(1-kappa+2*kappa*eta-kappa*eta^2)/(2*eta);



x = x0;
z = z0;
v = v0;
z_hat = z;
v_hat = v;
err = log(sqrt(x'*x + z'*z));
noise = 0;
dual_grad = [];
dual_grad_nominal = [];
for k=1:max_iter
    if k==1
        [x_next, z_next, v_next, z_hat_next,v_hat_next,err_next,noise_next]=...
            A_ADMM_update(z(:,end), v(:,end), z_hat(:,end), v_hat(:,end), ...
            k, M, N, rho,d);
    else
        [x_next, z_next, v_next, z_hat_next, v_hat_next, err_next, noise_next]=...
            A_ADMM_update3(x(:,end), v(:,end), z_hat(:,end), v_hat(:,end),...
            v(:,end-1), M, N, rho, alpha, beta, gamma,d);
    end
    dual_grad_nominal = [dual_grad_nominal, dual_star(v_hat(:,end), M,N, rho, d)];
    dual_grad = [dual_grad,[x_next;z_next]];
    
    x = [x,x_next];
    z = [z,z_next];
    v = [v,v_next];
    z_hat = [z_hat,z_hat_next];
    v_hat = [v_hat,v_hat_next];
    err = [err,err_next];
    noise = [noise,noise_next];
end


terms = ones(1,max_iter);
v_func = terms;
z_term = v_func;
q_k = terms;
real_terms = terms;
rr=terms;
e = ones(d,max_iter);
e_term = terms;
e_term_general = terms;
for j=1:max_iter
    rr(j) = norm(dual_grad(:,j)-dual_grad_nominal(:,j))/norm(dual_grad_nominal(:,j));
    
    %/(norm(dual_grad(:,j))+norm(dual_grad_nominal(:,j)))
    terms(j)=-(x(:,j+1) - z(:,j+1)- m*v_hat(:,j)  )'*(L*v_hat(:,j)-(x(:,j+1)-z(:,j+1)) );

    if j>=2
        e(:,j-1) = dual_grad(1:d,j-1)-dual_grad(d+1:end,j-1) - (dual_grad_nominal(1:d,j-1)-dual_grad_nominal(d+1:end,j-1));
        e_term(j-1) = e(:,j-1)'*(e(:,j-1) - (L*v_hat(:,j-1)-(dual_grad_nominal(1:d,j-1) - dual_grad_nominal(d+1:end,j-1))) ...
            +(dual_grad_nominal(1:d,j-1)-dual_grad_nominal(d+1:end,j-1) - m*v_hat(:,j-1)) );
        e_term_general(j-1) = norm(e(:,j-1))^2 - e_term(j-1);
        %(f(x(:,j+1),M)+g(z(:,j+1),N)+v_hat(:,j)'*(x(:,j+1)-z(:,j+1)) + rho/2*norm(x(:,j+1)-z(:,j+1))^2)
        %(f(x(:,j),M)+g(z(:,j),N)+v_hat(:,j-1)'*(x(:,j)-z(:,j)) + rho/2*norm(x(:,j)-z(:,j))^2)
        %dual_function(v_hat(:,j),M,N,rho,d)
        q_k(j-1) = (L-m)*( dual_function(v_hat(:,j),M,N,rho,d) ...
            + m/2*norm(v_hat(:,j-1))^2)...%grad_dual(v_hat(:,j),M,N,rho,d)
            +1/2*norm( grad_dual(v_hat(:,j),M,N,rho,d) -m*v_hat(:,j-1) )^2;%x(:,j+1)-z(:,j+1)
        %x(:,j)-z(:,j)
        z_term(j) = lambda*norm(1/(1-eta^2)*( v(:,j) - eta^2*v(:,j-1)))^2;
        v_func(j) = (1)*z_term(j) ...
            - q_k(j-1); % e_term_general(j-1)
        tt(j) = (x(:,j+1)-z(:,j+1) - m*v_hat(:,j))'*(L*(v_hat(:,j)-eta^2*v_hat(:,j-1) ) ...
            -  ((x(:,j+1)-z(:,j+1)) - eta^2* (x(:,j)-z(:,j))));
    end
end
term_qk = -q_k(4:end)+eta^2*q_k(3:end-1); % compare with tt

for j=3:max_iter
    real_terms(j) = -(x(:,j+1) - z(:,j+1)- m*v_hat(:,j)  )'...
        *(L*(v_hat(:,j)-eta^2*v_hat(:,j-1))-((x(:,j+1)-z(:,j+1))-eta^2*(x(:,j)-z(:,j))) );
    
end


figure
% plot(1:length(terms(1:end)),log(terms(1:end)))
v_function_value = log(v_func(6:end-1)+ v_func(5:end-2) + v_func(4:end-3) + v_func(3:end-4));
% % figure%+ v_func(6:end-1) + v_func(5:end-2) + v_func(4:end-3) +v_func(3:end-4))
plot(6:max_iter-1,v_function_value)% v_func(3:end-1)+v_func(4:end)
title('Lyapunov function candidate $$\tilde{V}_k=\sum^k_{i=k-4}V_i$$','interpreter','latex')
xlabel('iterations')
term = dual_grad - dual_grad_nominal;

figure
plot(6:max_iter-1,log(v_func(6:end-1)))% v_func(3:end-1)+v_func(4:end)
% % hold on
title('Lyapunov function candidate $$V_k$$','interpreter','latex')
xlabel('iterations')


rr_max = max(rr(4:end))







x_plot = 1:(length(err)-1);
y_plot = log(eta)*x_plot;

convergence_rate = exp((err(end) - err(end-20) )/(x_plot(end) - x_plot(end-20)))

% % figure
% plot(x_plot, err(2:end))
% hold on
% plot(x_plot, y_plot)
% legend('RA-ADMM','Predicted convergence rate')
% xlabel('iterations')
% ylabel('$$ \log(||x-x^*||^2 + ||z-z^*||^2) $$','interpreter','latex')


legend({'$$\eta = 1-1/\kappa_D$$',...
    '$$\eta = \frac{2}{3}(1-1/\kappa_D)+ \frac{1}{3}(1-1/\sqrt{\kappa_D})$$',...
    '$$\eta = \frac{1}{3}(1-1/\kappa_D)+ \frac{2}{3}(1-1/\sqrt{\kappa_D})$$',...
    '$$\eta = 1-1/\sqrt{\kappa_D}$$'},'interpreter','latex')
title('Robustness comparison under same level noise')
% -------------------------------------------------------------------------
function out = f(x,M)
out = 1/2 * x'*M*x;
end

function out = g(x,N)
out = 1/2 * x'*N*x;
end

function out = prox_f(z,v,M,rho,d)
out = (M + rho * eye(d))^-1 * (rho * z - v);
end

function out = prox_g(x,v,N,rho,d)
out = (N + rho * eye(d))^-1 * (rho * x + v);
end

% nominal from Rene Vidal
function [x_next, z_next, v_next, z_hat_next,v_hat_next,err_next,noise]=...
    A_ADMM_update(z, v, z_hat, v_hat, k, M, N, rho, d)
x_next = prox_f(z_hat, v_hat, M, rho, d);
z_next = prox_g(x_next, v_hat, N, rho, d);
noise = 0 * rand(1,1);
v_next = v_hat + (x_next - z_next)*(1 + noise);
v_hat_next = v_next + k/(k+3) * (v_next - v);
z_hat_next = z_next + k/(k+3) * (z_next - z);
err_next = log(sqrt(x_next'*x_next+z_next'*z_next));
end

function [x_next, z_next, v_next, z_hat_next, v_hat_next, err_next, noise]=...
    A_ADMM_update2(x,v,z_hat,v_hat,v_last, M, N, rho, alpha, beta, gamma,d)
x_next = prox_f(z_hat, v_hat, M, rho, d);
z_next = prox_g(x_next, v_hat, N, rho, d);
noise = 0 * rand(1,1);
v_next = v+ beta*(v - v_last) + alpha * (x_next - z_next)*(1+noise);
v_hat_next = v_next + gamma*(v_next - v);
z_hat_next = prox_g(x_next + gamma * (x_next - x),v_hat_next, N, rho, d);
err_next = log(sqrt(x_next'*x_next+z_next'*z_next));
end

% ultimate slim RA-ADMM:
function [x_next, z_next, v_next, z_hat_next, v_hat_next, err_next, noise]=...
    A_ADMM_update3(x,v,z_hat,v_hat,v_last, M, N, rho, alpha, beta, gamma,d)
noise = 0 * rand(1,1);

x_next = prox_f(z_hat, v_hat, M, rho, d);
z_next = prox_g(x_next, v_hat, N, rho, d);
v_next = v+ beta*(v - v_last) + alpha * (x_next - z_next)*(1+noise);
v_hat_next = v_next + gamma*(v_next - v);

z_hat_next = z_next;
% z_hat_next = prox_g(x_next + gamma * (x_next - x),v_hat_next, N, rho);

err_next = log(sqrt(x_next'*x_next+z_next'*z_next));
end

%  Why Rene's A-ADMM need to update v:
function [x_next, z_next, v_next, z_hat_next, v_hat_next, err_next, noise] = ...
    A_ADMM_update4(z,v,v_hat,M,N,rho,k,d)
x_next = prox_f(z,v_hat, M, rho, d);
z_next = prox_g(x_next,v_hat,N,rho, d);
v_next = v + (x_next - z_next);
v_hat_next = v_next + k/(k+3)*(v_next - v);

err_next = log(sqrt(x_next'*x_next+z_next'*z_next));
noise = 0;
z_hat_next = z_next;
end

function out = dual_function(u,M,N,rho,d)
I = eye(d);
sol = [M+rho*I, -rho*I;-rho*I,N+rho*I]^-1 *[-u;u];
x = sol(1:d);
z = sol((d+1): end);
out = f(x,M) + g(z,N) + u'*(x-z) + rho/2*norm(x-z)^2;
end

function out = grad_dual(u,M,N,rho, d)
I = eye(d);
sol = [M+rho*I, -rho*I;-rho*I, N+rho*I]^-1 * [-u;u];
out = sol(1:d)-sol((d+1):end);
end

function out = dual_star(u,M,N,rho,d)
I = eye(d);
out = [M+rho*I, -rho*I;-rho*I, N+rho*I]^-1 * [-u;u];
end