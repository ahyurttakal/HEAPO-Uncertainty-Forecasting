# Probabilistic metric definition

For quantile levels \(\mathcal{T}=\{0.10,0.50,0.90\}\), the implementation
first computes mean pinball loss

\[
L_{\tau}=\frac{1}{n}\sum_{i=1}^{n}
\max\left\{\tau(y_i-\hat q_{i,\tau}),
(\tau-1)(y_i-\hat q_{i,\tau})\right\}.
\]

It then reports

\[
\widetilde{\mathrm{CRPS}}_{0.10:0.90}
=2\,\operatorname{Trapz}_{\tau\in\{0.10,0.50,0.90\}}(L_{\tau}).
\]

This is a trapezoidal, three-quantile approximation over the observed
0.10–0.90 quantile range. It is not the full distributional CRPS, which would
require integration over the complete quantile function. Output columns retain
the explicit name `approx_crps`.

For the central 80% interval \([l_i,u_i]\), the interval score is

\[
\mathrm{IS}_{0.20}=(u_i-l_i)
+10(l_i-y_i)\mathbf{1}(y_i<l_i)
+10(y_i-u_i)\mathbf{1}(y_i>u_i).
\]

With one central interval and the median \(m_i\), the reported weighted
interval score is

\[
\mathrm{WIS}=\frac{1}{n}\sum_i
\frac{0.5|y_i-m_i|+0.1\,\mathrm{IS}_{0.20,i}}{1.5}.
\]

PICP, mean interval width, Winkler/interval score, and WIS are exported for
both raw and conformally calibrated intervals on the dense and common indices.
