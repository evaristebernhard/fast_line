import torch
import matplotlib.pyplot as plt
import math

# =========================
# 参数
# =========================
H = 2.0          # 总下降高度
X = 1.0          # 总水平距离
g = 9.81
n = 100          # 分成 100 段

dx = X / n

# Adam 实际优化的是 z_i
z = torch.zeros(n, dtype=torch.float64, requires_grad=True)

optimizer = torch.optim.Adam([z], lr=0.03)

# =========================
# 优化
# =========================
num_epochs = 10000

for epoch in range(num_epochs):

    optimizer.zero_grad()

    # 保证 q_i > 0
    q = torch.nn.functional.softplus(z)

    # 保证：
    # sum(s_i) = n H / X
    #
    # 因为 dy_i = s_i * dx
    # 所以 sum(dy_i) = H
    s = (n * H / X) * q / q.sum()

    # 每一段的垂直下降量
    dy = s * dx

    # y_0 = 0
    # y_i = sum_{j<=i} dy_j
    y_end = torch.cumsum(dy, dim=0)

    # 每段起点高度
    y_start = torch.cat([
        torch.zeros(1, dtype=torch.float64),
        y_end[:-1]
    ])

    # =========================
    # 每一小段的精确运动时间
    #
    # T_i =
    # 2 sqrt(1+s_i^2)/(s_i sqrt(2g))
    # * (sqrt(y_i)-sqrt(y_{i-1}))
    # =========================
    T_segments = (
        2.0
        * torch.sqrt(1.0 + s**2)
        / (s * math.sqrt(2.0 * g))
        * (torch.sqrt(y_end) - torch.sqrt(y_start))
    )

    T = T_segments.sum()

    T.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(
            f"epoch = {epoch:5d}, "
            f"T = {T.item():.8f} s, "
            f"y_end = {y_end[-1].item():.8f}"
        )


# =========================
# 得到最终路径
# =========================
with torch.no_grad():

    q = torch.nn.functional.softplus(z)

    s = (n * H / X) * q / q.sum()

    dy = s * dx

    y = torch.cat([
        torch.zeros(1, dtype=torch.float64),
        torch.cumsum(dy, dim=0)
    ])

    x = torch.linspace(
        0,
        X,
        n + 1,
        dtype=torch.float64
    )

    print("\nFinal:")
    print("sum(s_i) =", s.sum().item())
    print("target    =", n * H / X)

    print("final y =", y[-1].item())

    print("\n前 10 个斜率:")
    print(s[:10])

    print("\n后 10 个斜率:")
    print(s[-10:])


# =========================
# 求解析 cycloid，用来比较
# =========================

# 终点 theta 满足：
#
# X/H =
# (theta - sin(theta)) /
# (1 - cos(theta))
#
target_ratio = X / H

left = 1e-8
right = 2.0 * math.pi - 1e-8

for _ in range(100):
    mid = (left + right) / 2

    ratio = (
        mid - math.sin(mid)
    ) / (
        1.0 - math.cos(mid)
    )

    if ratio < target_ratio:
        left = mid
    else:
        right = mid

theta_end = (left + right) / 2

a = H / (1.0 - math.cos(theta_end))

theta = torch.linspace(
    0,
    theta_end,
    1000,
    dtype=torch.float64
)

x_exact = a * (
    theta - torch.sin(theta)
)

y_exact = a * (
    1.0 - torch.cos(theta)
)

print("\nCycloid parameters:")
print("theta_end =", theta_end)
print("a =", a)


# =========================
# 画图
# =========================
plt.figure(figsize=(8, 6))

plt.plot(
    x.numpy(),
    y.numpy(),
    "o-",
    markersize=2,
    label="Adam discrete path"
)

plt.plot(
    x_exact.numpy(),
    y_exact.numpy(),
    "--",
    linewidth=2,
    label="Exact cycloid"
)

plt.xlabel("x")
plt.ylabel("vertical drop y")
plt.title("Brachistochrone: Adam optimization")
plt.legend()
plt.grid()

# 让“向下”为视觉上的向下
plt.gca().invert_yaxis()

plt.show()


# =========================
# 画优化出来的 slope
# =========================
plt.figure(figsize=(8, 5))

plt.plot(
    x[:-1].numpy(),
    s.numpy()
)

plt.xlabel("x")
plt.ylabel("slope s_i")
plt.title("Optimized slope")
plt.grid()

plt.show()