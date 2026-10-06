# EC-SAV：目前可以严格写下来的理论

本文档针对代码中的模型

\[
 M\dot\omega+D\omega+\nabla U(\delta)=P,\qquad
 \dot\delta=\omega,
\]

其中 `M`、`D` 为正对角矩阵，且

\[
 H(\delta,\omega)=\frac12\omega^TM\omega+U(\delta)-P^T\delta .
\]

若 `P` 和 `U` 满足旋转不变性，则 `1^T P=0`、`1^T\nabla U=0`，连续系统满足

\[
 \dot H=-\omega^TD\omega .
\]

这里的结果是 EC-SAV prototype 的可证明性质和需要额外假设的地方，不是完整收敛定理，也不构成“文献首创”的结论。

## 1. SAV provisional step

取

\[
 r=\sqrt{U(\delta)+C},\qquad C> -\min U,
\]

并在预测点

\[
 \bar\delta^{n+1/2}=\delta^n+\frac h2\omega^n
\]

定义

\[
 b=\frac{\nabla U(\bar\delta^{n+1/2})}
          {\sqrt{U(\bar\delta^{n+1/2})+C}}.
\]

令

\[
 v=\frac{\omega^{n+1}-\omega^n}{2}+\omega^n
      =\frac{\omega^{n+1}+\omega^n}{2},
\qquad
 \delta^{n+1}=\delta^n+h v,
\]

使用

\[
 \frac{r^{n+1}-r^n}{h}=\frac12b^Tv,
\]

则速度方程化为

\[
 \left(\frac{2M}{h}+D+\frac h4bb^T\right)v
 =\frac{2M}{h}\omega^n+P-r^n b.
\]

因此当 `M,D` 对角时，矩阵是对角矩阵加 rank-one 更新。代码中的 Sherman--Morrison 实现是精确的线性代数求解，不是近似预条件器。

## 2. 修改能量耗散律

定义

\[
 \widetilde H^n
 =\frac12(\omega^n)^TM\omega^n+(r^n)^2-C-P^T\delta^n.
\]

将速度方程左乘 `h v^T`，再使用

\[
 (\omega^{n+1}-\omega^n)^TMv
 =\frac12\left[(\omega^{n+1})^TM\omega^{n+1}
              -(\omega^n)^TM\omega^n\right]
\]

以及

\[
 P^Tv=\frac{P^T(\delta^{n+1}-\delta^n)}h,
\qquad
 (r^{n+1})^2-(r^n)^2=h b^Tv(r^{n+1}+r^n),
\]

得到

\[
 \boxed{\widetilde H^{n+1}-\widetilde H^n
       =-h(v^{n+1/2})^TDv^{n+1/2}}.
\]

这解释了普通 SAV 的优点，也精确指出了它的问题：该恒等式不包含
`U(delta^{n+1})`，除非 `r^{n+1}` 恰好等于真实平方根。

## 3. EC correction 的存在性

对 provisional 位置 `delta_tilde` 和速度 `omega_tilde`，定义

\[
 c=\frac{\mathbf1^TM\widetilde\omega}{\mathbf1^TM\mathbf1}\mathbf1,
 \qquad q=\widetilde\omega-c,
\]

于是 `1^T M q=0`。校正限制为

\[
 \omega^{n+1}=c+s q.
\]

记

\[
 V=U(\widetilde\delta)-P^T\widetilde\delta,
 \quad z=\omega^n+c,
\]

并要求离散物理能量满足

\[
 H^{n+1}-H^n=-h\left(\frac{\omega^n+\omega^{n+1}}2\right)^T
 D\left(\frac{\omega^n+\omega^{n+1}}2\right).
\]

代入后得到

\[
 As^2+Bs+C_q=0,
\]

其中

\[
 A=\frac12q^TMq+\frac h4q^TDq,
\]

\[
 B=c^TMq+\frac h2z^TDq,
\]

\[
 C_q=\frac12c^TMc+V-H^n+\frac h4z^TDz.
\]

在理想的 `c` 定义下 `c^TMq=0`，但代码保留这一项以便以后替换约束。只要

\[
 A>0,\qquad \Delta_s=B^2-4AC_q\ge0,
\]

就存在实数校正根；若至少有一个根满足 `s>=0`，即可保持 provisional 相对速度方向。代码选择距离 `1` 最近的可接受根，这个选择保证小步长极限下选择连续分支，而不会随意发生符号翻转。

## 4. 无阻尼情形的严格结论

当 `D=0` 时，

\[
 A=\frac12q^TMq,\qquad B=0,
\]

所以

\[
 s^2=\frac{H^n-V-\frac12c^TMc}{\frac12q^TMq}.
\]

若右端非负，则 EC-SAV 输出满足

\[
 \boxed{H^{n+1}=H^n}
\]

到浮点舍入误差。并且由于 `1^TMq=0`，有

\[
 \mathbf1^TM\omega^{n+1}
 =\mathbf1^TM\widetilde\omega.
\]

对无阻尼、`1^TP=0` 的平移不变系统，SAV provisional step 本身保持总角动量；因此 EC 校正也保持同一个离散角动量。

这个结论是“在 correction admissible 时”的条件结论。不能把它写成无条件稳定性定理，因为当 `q=0` 或目标能量低于固定中心动能与势能时，速度方向缩放没有可行解。

## 5. 有阻尼情形的边界

对任意对角 `D>=0`，若二次方程有可接受根，则代码输出满足真实物理能量平衡

\[
 \boxed{H^{n+1}-H^n
 =-h\left(\frac{\omega^n+\omega^{n+1}}2\right)^T
 D\left(\frac{\omega^n+\omega^{n+1}}2\right)}.
\]

但“保持 provisional 总角动量”与阻尼下的精确动量方程一般不是同时成立的。若 `D=\gamma M`，则

\[
 \mathbf1^TDq=\gamma\mathbf1^TMq=0,
\]

所以这个额外约束与均匀比例阻尼相容；对一般非比例 `D`，EC-SAV 当前实现优先保持能量律，不能同时宣称精确保持阻尼动量律。

## 6. 二阶精度应如何表述

在以下局部条件下：

1. `U` 至少三阶连续可微；
2. `U+C` 在轨道邻域内有正下界；
3. provisional 解、`q` 和校正根都保持有界；
4. 所选根是靠近 `s=1` 的连续分支，并且 `A` 不退化；
5. 离散解不穿越故障切换时刻（切换用精确的子步结束），

预测点误差为 `O(h^2)`，SAV-CN provisional step 的局部误差为 `O(h^3)`。若 provisional 真实能量缺陷为 `O(h^3)` 且 `q^TMq` 有正下界，则隐函数/二次根连续性给出

\[
 s-1=O(h^3),
\]

因而 EC 校正不改变二阶方法的局部阶，整体仍为二阶。这个论证在 `q -> 0` 附近不一致；这正是需要单独研究的 turning-point / low-relative-kinetic-energy 情形。

## 7. 故障切换的能量映射

网络阶段 `k` 有自己的势能 `U_k` 和 Hamiltonian `H_k`。在理想三相故障/切线模型里，`delta` 和 `omega` 连续，但网络参数瞬时改变，因此

\[
 H_{k+1}(t_s^+)-H_k(t_s^-)
 =U_{k+1}(\delta(t_s))-U_k(\delta(t_s)).
\]

所以切换时不能继续沿用旧的 `Htarget`，也不能把 `r` 当作连续量。正确的事件映射是

\[
 r(t_s^+)=\sqrt{U_{k+1}(\delta(t_s))+C},
\]

\[
 H_{k+1}(t_s^+)=
 \frac12\omega(t_s)^TM\omega(t_s)
 +U_{k+1}(\delta(t_s))-P^T\delta(t_s).
\]

`ieee9_ecsav_cct.py` 按这个规则处理 fault -> post 切换。

## 8. 当前最重要的可证伪测试

下一轮不应只报告 RMSE。建议固定以下诊断量：

* `max |H^{n+1}-H^n|`（无阻尼）；
* `max |H^{n+1}-H^n+h v^TDv|`（有阻尼）；
* correction discriminant 的最小值；
* `min/max s` 及 correction failure 次数；
* fault/post 切换前后势能跳变是否等于 `U_post-U_fault`；
* `|CCT_h-CCT_ref|` 和 stable/unstable 二分类是否翻转。

尤其要把 correction failure 当成结果报告：如果 `A<=0`、判别式为负或 `q` 退化，EC-SAV 需要明确的 fallback / step rejection 策略，这将是完整算法设计的一部分，而不是实现细节。
