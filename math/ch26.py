from _lab_support import begin, budget, write_result
LAB_MODE = begin('26')
# Experiment M26; independent synthetic method demonstration, not main case A.
import numpy as np, warnings
warnings.filterwarnings("ignore")
np.set_printoptions(precision=5, suppress=True, linewidth=160)
def sigmoid(z): return 1/(1+np.exp(-z))

# ---------- A: 2-2-1 MLP, ReLU hidden, sigmoid output, cross-entropy ----------
x = np.array([1.0, 2.0]); y = 1.0
W1 = np.array([[0.2, 0.4],[0.3,-0.1]]); b1 = np.array([0.1, 0.2])
w2 = np.array([0.6,-0.4]); b2 = 0.1
def forward(W1,b1,w2,b2,x):
    z1 = W1@x + b1; h = np.maximum(z1,0); a = w2@h + b2; p = sigmoid(a); return z1,h,a,p
def loss(W1,b1,w2,b2,x,y):
    p = forward(W1,b1,w2,b2,x)[3]; return -(y*np.log(p)+(1-y)*np.log(1-p))
z1,h,a,p = forward(W1,b1,w2,b2,x)
print("== A: forward ==")
print(" z1 =",z1," h =",h," a =",a," p =",p," loss =",loss(W1,b1,w2,b2,x,y))
d_out = p - y
g_w2 = d_out*h; g_b2 = d_out
d_h = (w2*d_out)*(z1>0)
g_W1 = np.outer(d_h, x); g_b1 = d_h
print(" backward: delta_out = p-y =", d_out)
print("  dL/dw2 =", g_w2, " dL/db2 =", g_b2)
print("  delta_h =", d_h)
print("  dL/dW1 =\n", g_W1, "\n  dL/db1 =", g_b1)
# finite-difference check
def num_grad(f, arr, eps=1e-6):
    g=np.zeros_like(arr); it=np.nditer(arr, flags=['multi_index'])
    for _ in it:
        i=it.multi_index; old=arr[i]
        arr[i]=old+eps; fp=f(); arr[i]=old-eps; fm=f(); arr[i]=old
        g[i]=(fp-fm)/(2*eps)
    return g
f=lambda: loss(W1,b1,w2,b2,x,y)
print(" finite-difference check (max abs diff):",
      np.abs(num_grad(f,W1)-g_W1).max(), np.abs(num_grad(f,b1)-g_b1).max(), np.abs(num_grad(f,w2)-g_w2).max(), abs(num_grad(f,np.array([b2]))[0]-g_b2) if False else "")
b2arr=np.array([b2]);
def f_b2(): return loss(W1,b1,w2,b2arr[0],x,y)
print("  b2 diff:", abs(num_grad(f_b2,b2arr)[0]-g_b2))
# one SGD step, lr=0.5, with and without weight decay lam=0.1 (on weights, not biases)
lr=0.5
W1n=W1-lr*g_W1; b1n=b1-lr*g_b1; w2n=w2-lr*g_w2; b2n=b2-lr*g_b2
print(" after 1 SGD step (lr=0.5): loss =", loss(W1n,b1n,w2n,b2n,x,y), " p =", forward(W1n,b1n,w2n,b2n,x)[3])
lam=0.1
W1d=W1-lr*(g_W1+lam*W1); w2d=w2-lr*(g_w2+lam*w2)
print(" with L2 weight decay 0.1 on weights: W1 =\n", W1d, "\n  w2 =", w2d, " loss after step =", loss(W1d,b1n,w2d,b2n,x,y))
print(" weight-decay multiplicative form: (1 - lr*lam) =", 1-lr*lam)

# ---------- B: no hidden layer = logistic regression; gradient = -(score) ----------
print("\n== B: 0 hidden layers = logistic regression ==")
xx = np.r_[np.zeros(10), np.ones(10)]; yy = np.r_[np.r_[np.ones(2),np.zeros(8)], np.r_[np.ones(6),np.zeros(4)]]
Xd = np.c_[np.ones(20), xx]; beta=np.zeros(2); pp=sigmoid(Xd@beta)
print(" gradient of summed cross-entropy at beta=0:", Xd.T@(pp-yy), " (= minus the score (-2, 1) of Chapter 21)")

# ---------- C: parameter counts ----------
def mlp_params(sizes): return sum((a+1)*b for a,b in zip(sizes[:-1],sizes[1:]))
print("\n== C: parameter counts ==")
for sizes in ([20,1],[20,64,64,1],[18,8,1],[18,64,64,1],[20,16,1]):
    n=mlp_params(sizes); print(f"  {sizes}: {n} parameters ; per event (160 events): {n/160:.2f}")

# ---------- D: early stopping vs ridge, 1-d quadratic ----------
print("\n== D: gradient flow (early stopping) vs ridge, shrink factor of a direction with curvature d ==")
print("   s=d*t   GD-path 1-exp(-s)   ridge s/(1+s)   (ridge lambda = 1/t)")
for s in (0.1,0.5,1,2,5): print(f"   {s:5.1f}   {1-np.exp(-s):.4f}            {s/(1+s):.4f}")
# discrete GD: eps=0.1, d=1
print(" discrete GD eps=0.1, d=1:  tau, 1-(1-eps d)^tau")
for tau in (5,10,20,50): print(f"   tau={tau:3d}: {1-(1-0.1)**tau:.4f}   (s={0.1*tau})")

# Repository acceptance checks use exact identities where possible.
write_result('26', {'backprop_matches_finite_difference': np.abs(num_grad(f,W1)-g_W1).max() < 1e-8 and abs(num_grad(f_b2,b2arr)[0]-g_b2) < 1e-8, 'sgd_reduces_loss': loss(W1n,b1n,w2n,b2n,x,y) < loss(W1,b1,w2,b2,x,y), 'parameter_count_20_64_64_1': mlp_params([20,64,64,1]) == 5569}, {'max_gradient_error': float(np.abs(num_grad(f,W1)-g_W1).max()), 'parameters_20_64_64_1': mlp_params([20,64,64,1])})
