import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


class wind_data_generator:
    def __init__(self, mode='train', normalize='0-1'):
        self.mode = mode
        self.gap = 1
        ds = xr.open_dataset('/mnt/A/era5/U_wind_500hPa/download.nc')
        if self.mode == 'train':
            train_ds = ds.sel(time=slice('2015', '2017'))
            train_data_max = train_ds.u.max().compute().values
            train_data_min = train_ds.u.min().compute().values

            self.n_samples = len(train_ds['time'].values)

            self.param = {'lon': np.array(train_ds['longitude'].values), 'lat': np.array(train_ds['latitude'].values)}
            self.data = (train_ds - train_data_min) / (train_data_max - train_data_min)
            self.train_ds_samples = len(train_ds['time'].values)

        elif self.mode == 'valid':
            self.valid_ds = ds.sel(time=slice('2018', '2018'))
            self.valid_data_min = self.valid_ds.u.min().compute().values
            self.valid_data_max = self.valid_ds.u.max().compute().values
            self.data = (self.valid_ds - self.valid_data_min)/(self.valid_data_max - self.valid_data_min)
            self.param = {'lon': np.array(self.valid_ds['longitude'].values),
                          'lat': np.array(self.valid_ds['latitude'].values)}
            self.valid_samples = len(self.valid_ds['time'].values)
            # print(self.valid_ds)
            self.valid_ds = ds.isel(expver=0)
            # print(self.valid_ds)
        print('0.25deg_U_Wind_getdata')

    def get_train_batch(self, input_length, output_length,
                        batch_size, item=0):
        in_len = input_length
        out_len = output_length
        item = item * batch_size
        batch_x = []
        batch_y = []
        batch_y_time = []
        batch_x_time = []
        for j in range(batch_size):
            x = []
            y = []
            x_time = []
            y_time = []
            for i in range(in_len):
                x.append(self.data.u.isel(time=item + i * self.gap, expver=0).values)
                x_time.append(self.data['time'][item + i * self.gap].values)
            for k in range(out_len):
                y.append(self.data.u.isel(time=item + k * self.gap + in_len * self.gap, expver=0).values)
                y_time.append(self.data['time'][item + k * self.gap + in_len * self.gap].values)
            batch_x.append(x)
            batch_y.append(y)
            batch_x_time.append(x_time)
            batch_y_time.append(y_time)
            item = item + 1
            # for t in x_time:
            #     print(t)
            # print('===================================================')
            # for s in y_time:
            #     print(s)
        x = np.array(batch_x)
        y = np.array(batch_y)
        x_time = np.array(batch_x_time)
        y_time = np.array(batch_y_time)

        return x, y, x_time, y_time

    def get_valid_batch(self, input_length, output_length,
                        batch_size, item=0):
        in_len = input_length
        out_len = output_length
        batch_x = []
        batch_y = []
        batch_y_time = []
        batch_x_time = []
        for j in range(batch_size):
            x = []
            y = []
            x_time = []
            y_time = []
            for i in range(in_len):
                x.append(self.data.u.isel(time=item + i * self.gap, expver=0).values)
                x_time.append(self.data['time'][item + i * self.gap].values)
            for j in range(out_len):
                y.append(self.data.u.isel(time=item + j * self.gap + in_len * self.gap, expver=0).values)
                y_time.append(self.data['time'][item + j * self.gap + in_len * self.gap ].values)
            batch_x.append(x)
            batch_y.append(y)
            batch_x_time.append(x_time)
            batch_y_time.append(y_time)
            # for t in x_time:
            #     print(t)
            # print('--------------------------------------------------')
            # for s in y_time:
            #     print(s)
        x = np.array(batch_x)
        y = np.array(batch_y)
        x_time = np.array(batch_x_time)
        y_time = np.array(batch_y_time)
        return x, y, x_time, y_time

# if __name__ == '__main__':
#     # train_data_generator = wind_data_generator()
#     # x,y,time1,time = train_data_generator.get_train_batch(input_length=5,output_length=10,batch_szie=10)
#     valid_data_generator = wind_data_generator(mode='valid')
#     x,y,time1,time = valid_data_generator.get_valid_batch(input_length=5,output_length=10, batch_size=1)
