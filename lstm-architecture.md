# LSTM Architecture

MM 03/12/2018

![](/assets/schematic_cell_in_LSTM_separate_version.jpg)**Fig.1 Schematic of a cell in a LSTM.**

Fig.1 shows the schematic of a cell in a LSTM.

The

$$s_{ci,n}^{\ell,t} = \sum_{\ell_{ci}=1}^{N_{ci}^\ell} w_{\ell_{ci},n}^{ci,\ell} \times x_{\ell_{ci}}^{\ell,t} + \sum_{\ell_{ci}=1}^{N_{ci}^{-,\ell}} w_{\ell_{ci}^-,n}^{ci-,\ell} \times x_{\ell_{ci}}^{\ell+1,t-1} + \sum_{\ell_{cic}=1}^{N_{cic}^\ell} w_{\ell_{cic},n}^{cic,\ell} \times s_{c,n}^{\ell,t-1}(\ell_{cic})  \\=\bar{w}_n^{ci,\ell} \cdot \bar{x}^{\ell,t} + \bar{w}_n^{ci-,\ell} \cdot \bar{x}^{\ell+1,t-1} +  \bar{w}_n^{cic,\ell} \cdot \bar{s}_{c,n}^{\ell+1,t-1}$$

$$s_{ig,n}^{\ell,t} = \sum_{\ell_{ig}=1}^{N_{ig}^\ell} w_{\ell_{ig},n}^{ig,\ell} \times x_{\ell_{ig}}^{\ell,t} + \sum_{\ell_{ig}=1}^{N_{ig}^{-,\ell}} w_{\ell_{ig}^-,n}^{ig-,\ell} \times x_{\ell_{ig}}^{\ell+1,t-1} + \sum_{\ell_{igc}=1}^{N_{igc}^\ell} w_{\ell_{igc},n}^{igc,\ell} \times s_{c,1}^{\ell,t-1}(\ell_{igc}) \\ =\bar{w}_n^{ig,\ell} \cdot \bar{x}^{\ell,t} + \bar{w}_n^{ig-,\ell} \cdot \bar{x}^{\ell+1,t-1} +  \bar{w}_n^{igc,\ell} \cdot \bar{s}_{c,n}^{\ell+1,t-1}$$

$$s_{f,n}^{\ell,t} = \sum_{\ell_{f}=1}^{N_{f}^\ell} w_{\ell_{f},n}^{f,\ell} \times x_{\ell_{f}}^{\ell,t} + \sum_{\ell_{f}=1}^{N_{f}^{-,\ell}} w_{\ell_{f}^-,n}^{f-,\ell} \times x_{\ell_{f}}^{\ell+1,t-1} + \sum_{\ell_{fc}=1}^{N_{fc}^\ell} w_{\ell_{fc},n}^{fc,\ell} \times s_{c,n}^{\ell,t-1}(\ell_{fc}) \\ =\bar{w}_n^{f,\ell} \cdot \bar{x}^{\ell,t} + \bar{w}_n^{f-,\ell} \cdot \bar{x}^{\ell+1,t-1} +  \bar{w}_n^{fc,\ell} \cdot \bar{s}_{c,n}^{\ell+1,t-1}$$

$$s_{c,n}^{\ell,t} = f_f^\ell(s_{f,n}^{\ell,t}) \times s_{c,n}^{\ell,t-1} + g_{ci}^\ell(s_{ci,n}^{\ell,t}) \times f_{ig}^\ell (s_{ig,n}^{\ell,t})$$

$$s_{og,n}^{\ell,t} = \sum_{\ell_{f}=1}^{N_{f}^\ell} w_{\ell_{f},1}^{og,\ell} \times x_{\ell_{og}}^{\ell,t} + \sum_{\ell_{og}=1}^{N_{og}^{-,\ell}} w_{\ell_{og}^-,n}^{og-,\ell} \times x_{\ell_{og}}^{\ell+1,t-1} + \sum_{\ell_{ogc}=1}^{N_{ogc}^\ell} w_{\ell_{ogc},n}^{ogc,\ell} \times s_{c,n}^{\ell,t-1}(\ell_{ogc}) \\ =\bar{w}_n^{og,\ell} \cdot \bar{x}^{\ell,t} + \bar{w}_n^{og-,\ell} \cdot \bar{x}^{\ell+1,t-1} +  \bar{w}_n^{ogc,\ell} \cdot \bar{s}_{c,n}^{\ell+1,t-1} $$

$$x_n^{\ell+1,t} = h_{co}^\ell(s_{c,n}^{\ell,t}) \times f_{og}^\ell(s_{og,n}^{\ell,t}) , \hspace{0.1 in} x_n^{\ell+2,t} = \sum_{m=1}^{N_{\ell+1}} w_{m,n}^{\ell+1} \times x_m^{\ell+1} + b_n^{\ell+2}$$

$$s_n^{\ell+2,t} = \sum_{m=1}^{N_{\ell+1}} w_{m,n} \times x_m^{\ell+1} + b_n^{\ell+2}, \hspace{0.1 in} f_{\ell +2} (s_n^{\ell+2,t})=x_n^{\ell+2,t}$$

![](/assets/schematic_LSTM.jpg)

Fig.2 Schematic of LSTM network.

