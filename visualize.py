import xarray as xr
import matplotlib.pyplot as plt
import High_deg_ConvGRU_Encoder_Decoder_model as standard_gru
import torch

ds = xr.open_dataset('/mnt/A/era5/U_wind_500hPa/download.nc')
# a = 0
# for i in range(a, a + 16):
#     U_R = ds_R.u.isel(time=i, expver=0)
#     U_R.plot(vmin=-20, vmax=20, cmap='RdBu_r')
#     plt.show()

train_ds = ds.sel(time=slice('2015', '2017'))
train_data_max = train_ds.u.max().compute().values
train_data_min = train_ds.u.min().compute().values

n_samples = len(train_ds['time'].values)

print('train_max', train_data_max)
print('train_min', train_data_min)
print('n_samples', n_samples)
model = standard_gru.multilayers_Painter_model(out_length=10, in_length=5).cuda()
a = model.encoder_GRU.forward(torch.zeros(15, 5, 1, 101, 73).cuda())
b = model.decoder_GRU.forward(a)
print(b.shape)
c = model.decoder(b[:, 0])
print(c.shape)
# print(b[0].shape, b[1].shape, b[2].shape)
