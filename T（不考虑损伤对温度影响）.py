from subprocess import *
from dolfin import *
import numpy as np
import os, shutil
import time

start = time.time()

parameters["form_compiler"]["quadrature_degree"] = 1
parameters["form_compiler"]["cpp_optimize"] = True

# 网格
mesh = Mesh('Temp.xml')

# 函数空间的定义
U = VectorFunctionSpace(mesh, 'CG', 1)
D = FunctionSpace(mesh, 'CG', 1)

# 对边进行标记
remain_b = CompiledSubDomain("x[1] <= 0 + tol", tol=1E-14)
remain_u = CompiledSubDomain("x[1] >= 0.005 - tol", tol=1E-14)
remain_l = CompiledSubDomain("x[0] <= 0 + tol", tol=1E-14)
remain_r = CompiledSubDomain("x[0] >= 0.025 - tol", tol=1E-14)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  # 标记网格的边界
boundaries.set_all(0)  # 将网格的所有边界初始化为0
remain_b.mark(boundaries, 1)
remain_u.mark(boundaries, 2)
remain_l.mark(boundaries, 3)
remain_r.mark(boundaries, 4)
ds = Measure("ds", subdomain_data=boundaries)  # ds表示积分测度，表示Fencis进行积分只会对boundaries标记为1的域进行积分
dx = dx(metadata={'quadrature_degree': 1})  # quadrature order

# 材料参数
E = 3.7e11
nu = 0.3
ft = 1.8E8
Gf = 42.47
b = 1.5E-4
# 线性软化 参数
k0 = -1.3546 * ft ** 2 / Gf
xwc = 5.1361 * Gf / ft
lch = E * Gf / ft ** 2
a1 = 5.5418
a2 = -0.5
a3 = 0
lamda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
thickness = 0.001
# 温度参数
rho = 3980
c = 880
k = 31
T_final = 293.15
T_0 = 573.15


# 相关函数定义
# 应变张量
def eps(u):
    return sym(grad(u))


# 热膨胀应变
def epsth(T_):
    return 7.5e-6 * (T_ - 573.15) * Identity(2)


# 有效应力张量
def sig0(u, T_):
    return lamda * tr(eps(u) - epsth(T_)) * Identity(2) + 2 * mu * (eps(u) - epsth(T_))


def sigma(u, d, T_):
    return w(d) * sig0(u, T_)  # 受损材料应力张量


# 驱动力
def Y(u, T_):
    s1 = (sig0(u, T_)[0, 0] + sig0(u, T_)[1, 1]) / 2. \
         + sqrt(((sig0(u, T_)[0, 0] - sig0(u, T_)[1, 1]) / 2.) ** 2 + sig0(u, T_)[0, 1] ** 2)
    s2 = (sig0(u, T_)[0, 0] + sig0(u, T_)[1, 1]) / 2. \
         - sqrt(((sig0(u, T_)[0, 0] - sig0(u, T_)[1, 1]) / 2.) ** 2 + sig0(u, T_)[0, 1] ** 2)
    smax = conditional(gt(s1, s2), s1, s2)
    sigeq = (smax + abs(smax)) / 2
    return sigeq ** 2 / 2. / E


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


num_steps = 1000
total_disp = 1
dt = total_disp / num_steps

# 定义边界条件
tol = 1E-6


def boundary_up(x, on_boundary):
    return x[1] >= 0.005 - tol


bcup = DirichletBC(D, Constant(T_final), boundary_up)


def boundary_l(x, on_boundary):
    return x[0] <= 0 + tol


bcdo = DirichletBC(D, Constant(T_final), boundary_l)


def boundary_r(x, on_boundary):
    return x[0] >= 0.025 - tol


bcr = DirichletBC(U.sub(0), Constant(0.0), boundary_r, method='pointwise')


def boundary_b(x, on_boundary):
    return x[1] <= 0 + tol


bcb = DirichletBC(U.sub(1), Constant(0.0), boundary_b, method='pointwise')

bct = [bcup, bcdo]
bcu = [bcr, bcb]

# 定义测试函数和试探函数
T = TrialFunction(D)
v = TestFunction(U)
q = TestFunction(D)

# 定义函数
u = Function(U)
d = Function(D)
T_ = Function(D)
T__ = Function(D)
n = FacetNormal(mesh)
T0 = Expression('T_0', T_0=T_0, degree=1)
T_ = interpolate(T0, D)

Eu = inner(w(d) * sig0(u, T_), eps(v)) * dx  # 位移场控制方程
Euu = derivative(Eu, u)
Ed = (2. * Gf / pi / b * ((1. - d) * q + b ** 2 * dot(grad(d), grad(q))) + wp(d) * Y(u, T_) * q) * dx  # 相场控制方程
Edd = derivative(Ed, d)

u.rename('displacement', 'displacement')  # 更改变量名
d.rename('damage', 'damage')
T_.rename('T', 'T')

# 温度场变分问题
F = rho * c * (T - T_) / dt * q * dx - k * dot(grad(T), n) * q * (ds(2) + ds(3)) + k * dot(grad(T), grad(q)) * dx
a = lhs(F)
L = rhs(F)


def T_solver(a, L, T_):
    A1 = assemble(a)
    L1 = assemble(L)
    [bc.apply(A1) for bc in bct]
    [bc.apply(L1) for bc in bct]
    solve(A1, T__.vector(), L1)
    T_.assign(T__)


# 相场位移场的耦合
# 位移控制方程的变分问题
problem_u = NonlinearVariationalProblem(Eu, u, bcu, Euu)
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


savedir = "results0/"
if os.path.isdir(savedir):
    shutil.rmtree(savedir)
file_alpha = File(savedir + "/alpha.pvd")
file_u = File(savedir + "/u.pvd")
file_T = File(savedir + "/T.pvd")


def postprocessing():
    if n % 1 == 0:
        file_alpha << (d, n)  # Phase field
        file_u << (u, n)  # Displacement field
        file_T << (T_, n)  # Displacement field


for n in range(num_steps):
    T_solver(a, L, T_)
    alternate(u, d, maxiter=50000)  # call solver function
    postprocessing()  # 记录位移荷载曲线
    print("-----------------------------------------")
    lb.vector()[:] = d.vector()  # updating the lower bound to account for the irreversibility(相场的不可逆）

# Print time taken to complete the simulation
end = time.time()
print(f"Runtime of the program is {end - start} sec")
