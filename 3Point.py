from subprocess import *
from dolfin import *
import numpy as np
import os, shutil
import time
import matplotlib.pyplot as plt

start = time.time()

parameters["form_compiler"]["quadrature_degree"] = 1
parameters["form_compiler"]["cpp_optimize"] = True

# 网格
mesh = Mesh('3point.xml')
plot(mesh)
plt.show()

# 函数空间的定义
U = VectorFunctionSpace(mesh, 'CG', 1)
D = FunctionSpace(mesh, 'CG', 1)

# 对底边标记为1
remain = CompiledSubDomain("x[1] >= 103-tol and x[0] > 222.5-tol and x[0] < 227.5 +tol",
                           tol=1E-14)  # 定义为L板的底边,tol为计算机精度的容差
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  # 标记网格的边界
boundaries.set_all(0)  # 将网格的所有边界初始化为0
remain.mark(boundaries, 1)  # 将底边标记为1
ds = Measure("ds", subdomain_data=boundaries)  # ds表示积分测度，表示Fencis进行积分只会对boundaries标记为1的域进行积分
dx = dx(metadata={'quadrature_degree': 1})  # quadrature order

# 材料参数
E = 2E4
nu = 0.2
ft = 2.4
Gf = 0.113
b = 2.5

# cornelissen 参数
k0 = -1.3546 * ft ** 2 / Gf
xwc = 5.1361 * Gf / ft
lch = E * Gf / ft ** 2
a1 = 4. / pi * lch / b
a2 = 2. * (-2 * k0 * Gf / ft ** 2) ** (2 / 3) - (2 + 0.5)

a3 = 1. / a2 * (0.125 * (xwc * ft / Gf) ** 2 - (1 + a2))
lamda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
thickness = 100


# 相关函数定义
# 应变张量
def eps(u):
    return sym(grad(u))


# stress tensor
def sig0(u):
    return lamda * tr(eps(u)) * Identity(2) + 2 * mu * eps(u)


def sigma(u, d):
    return w(d) * sig0(u)  # 受损材料应力张量


def Y(u):
    s1 = (sig0(u)[0, 0] + sig0(u)[1, 1]) / 2. \
         + sqrt(((sig0(u)[0, 0] - sig0(u)[1, 1]) / 2.) ** 2 + sig0(u)[0, 1] ** 2)
    s2 = (sig0(u)[0, 0] + sig0(u)[1, 1]) / 2. \
         - sqrt(((sig0(u)[0, 0] - sig0(u)[1, 1]) / 2.) ** 2 + sig0(u)[0, 1] ** 2)
    s3 = lamda / 2. / (lamda + mu) * (s1 + s2)
    smax = conditional(gt(s1, s2), s1, s2)
    betac = 9.
    J2bar = .5 * (s1 ** 2 + s2 ** 2 + s3 ** 2 - (s1 + s2 + s3) ** 2 / 3.)
    sigeq = (smax + abs(smax)) / 2
    # sigeq = (betac * (smax + abs(smax)) / 2. \
    #           + sqrt(3. * J2bar)) / (1. + betac)
    return sigeq ** 2 / 2. / E


num_steps = 2000
total_disp = -1.0  # total applied displacement（总的加载位移）


# 退化函数
def w(d):
    R = (1.0 - d) ** 2
    Q = a1 * d + a1 * a2 * d ** 2.0 + a1 * a2 * a3 * d ** 3.0
    return R / (R + Q)


def wp(d):
    R = (1. - d) ** 2
    Rp = -2 * (1. - d) ** (2 - 1.)
    Q = a1 * d + a1 * a2 * d ** 2. + a1 * a2 * a3 * d ** 3.
    Qp = a1 + 2. * a1 * a2 * d + 3. * a1 * a2 * a3 * d ** 2.
    return (Q * Rp - R * Qp) / (R + Q) ** 2.


# 定义边界条件
tol = 1E-6


# 左支座
def fix(x, on_boundary):
    return x[1] <= 0.0 + tol and x[0] <= 0 + tol  # 返回一个布尔值，判断是否在边界上


# 右支座
def roller(x, on_boundary):
    return x[0] >= 450 - tol and x[1] <= 0.0 + tol  # 同上


# Location to apply displacement（位移加载的位置）
def disp(x, on_boundary):
    return x[1] >= 103 - tol and x[0] >= 224.9 - tol and x[0] <= 225.1 + tol  # 同上


bcL = DirichletBC(U, Constant((0.0, 0.0)), fix, method='pointwise')  # define boundary condition
# 定义狄利克雷边界：定义为位移的第二个分量y，位移为0，boundary_D_b为刚刚定义的底边，点值方式施加
bcR = DirichletBC(U.sub(1), Constant(0), roller, method='pointwise')  # define boundary condition
u_R = Expression(('disp_A + disp_app*(n+1-B)'), disp_A=0.0, disp_app=total_disp / num_steps, n=0., B=0.,
                 degree=0)  # Define loading as an expression so that it can be updated for next step
bcdisp = DirichletBC(U.sub(1), u_R, disp, method='pointwise')
bc_disp = [bcL, bcR, bcdisp]

# 定义测试函数和试探函数
v = TestFunction(U)
u_ = TrialFunction(U)
q = TestFunction(D)
d_ = TrialFunction(D)

# 定义函数
u = Function(U)
d = Function(D)
Eu = inner(w(d) * sig0(u), eps(v)) * dx  # 位移场控制方程
Euu = derivative(Eu, u, u_)
Ed = (2. * Gf / pi / b * ((1. - d) * q + b ** 2 * dot(grad(d), grad(q))) + wp(d) * Y(u) * q) * dx  # 相场控制方程
Edd = derivative(Ed, d, d_)

u.rename('displacement', 'displacement')  # 更改变量名
d.rename('damage', 'damage')

# 位移控制方程的变分问题
problem_u = NonlinearVariationalProblem(Eu, u, bc_disp, Euu)
solver_u = NonlinearVariationalSolver(problem_u)
prmu = solver_u.parameters
prmu["newton_solver"]["relative_tolerance"] = 1E-3
prmu["newton_solver"]["absolute_tolerance"] = 1E-3
prmu["newton_solver"]["convergence_criterion"] = "residual"  # 残差
prmu["newton_solver"]["error_on_nonconvergence"] = True  # 无法收敛时报告错误
prmu["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
prmu["newton_solver"]["linear_solver"] = 'mumps'  # 线性求解器类型
prmu["newton_solver"]["lu_solver"]["symmetric"] = False  # LU不对称
prmu["newton_solver"]["maximum_iterations"] = 10000
prmu["newton_solver"]["relaxation_parameter"] = 1.0

# 相场控制方程的变分问题
problem_d = NonlinearVariationalProblem(Ed, d, [], Edd)
lb = interpolate(Constant(0), D)
ub = interpolate(Constant(1), D)
solver_d = NonlinearVariationalSolver(problem_d)
snes_solver_prm = {"nonlinear_solver": "snes",
                   "snes_solver": {"maximum_iterations": 500,
                                   "report": False,
                                   "linear_solver": "umfpack",
                                   "preconditioner": "bjacobi",
                                   "line_search": "basic",
                                   "method": "vinewtonrsls",
                                   "absolute_tolerance": 1e-8,
                                   "relative_tolerance": 1e-8,
                                   "krylov_solver": {"absolute_tolerance": 1e-8,
                                                     "relative_tolerance": 1e-8}}}
solver_d.parameters.update(snes_solver_prm)


def alternate(u, d, tol=5e-4, maxiter=10000, d_0=interpolate(Constant("0.0"), D)):
    iter = 1
    err_alpha = 1
    alpha_error = Function(D)
    while err_alpha > tol and iter < maxiter:
        # solve elastic problem
        print('solve displacement')
        solver_u.solve()
        print('solve  phase field')
        # solve phase field problem
        problem_d.set_bounds(lb.vector(), ub.vector())
        solver_d = NonlinearVariationalSolver(problem_d)
        solver_d.parameters.update(snes_solver_prm)
        solver_d.solve()
        alpha_error.vector()[:] = d.vector() - d_0.vector()  # 计算相场误差
        err_alpha = np.linalg.norm(alpha_error.vector().get_local(), ord=np.Inf)  # 获取误差，并计算范数
        # update iteration
        print("Iteration:  %2d, Error: %2.8g, alpha_max: %.8g" % (iter, err_alpha, d.vector().max()))
        d_0.assign(d)
        iter = iter + 1
    print("Iteration:  %2d, Error: %2.8g, alpha_max: %.8g" % (iter - 1, err_alpha, d.vector().max()))
    return (err_alpha, iter)


savedir = "results_3p/"
if os.path.isdir(savedir):
    shutil.rmtree(savedir)
file_alpha = File(savedir + "/alpha.pvd")
file_u = File(savedir + "/u.pvd")
forces = np.zeros((num_steps + 1, 2))


def postprocessing():
    forces[n + 1] = np.array([-u(225, 103)[1], -assemble(sigma(u, d)[1, 1] * thickness * ds(1))])
    # Dump solution to file
    if n % 10 == 0:
        file_alpha << (d, n)  # Phase field
        file_u << (u, n)  # Displacement field
    np.savetxt(savedir + '/forces_3p.txt', forces)  # record force displacement data


# Execution of the loading steps
for n in range(num_steps):
    u_R.n = n
    # solve alternate minimization
    alternate(u, d, maxiter=50000)  # call solver function

    postprocessing()  # 记录位移荷载曲线
    print("\nEnd of timestep %d with load %g" % (n, -u(225, 103.)[1]))  # print completion of load step in terminal
    print("-----------------------------------------")
    lb.vector()[:] = d.vector()  # updating the lower bound to account for the irreversibility(相场的不可逆）

# Print time taken to complete the simulation
end = time.time()
print(f"Runtime of the program is {end - start} sec")
