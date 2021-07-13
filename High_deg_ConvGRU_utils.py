import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
from skimage.measure import compare_mse     #均方误差
from skimage.measure import compare_psnr    #峰值信噪比
from skimage.measure import compare_ssim    #结构相似性
print('0.25deg_U_Wind_util')


def save_x_img(idx,dataset,x_data_time,label,
               path='/extend/A/weather-benchmark/10m_U_component_wind/wind_prediction/picture',
               in_len=5, out_len=10):
    path = path + '/valid' + str(idx)
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    if label == 'input':
        input_path = path+'/'+'input'
        len = in_len
    else:
        input_path = path+'/'+'output'
        len = out_len
    if os.path.exists(input_path):
        pass
    else:
        os.mkdir(input_path)
    for t_idx in range(len):
        time = x_data_time[t_idx]
        t = str(time)
        x_data = dataset.u.sel(time=slice(t, t))
        plt.figure(figsize=(10, 10))
        x_data.plot(vmin=-15, vmax=15,cmap='RdBu_r')
        # x_data.plot(cmap='RdBu_r')
        plt.savefig(input_path + '/' + str(t) + '.jpg')
        plt.close()


def MNGO_save2jpg(idx,recons,y_data_time,param,min, max,
                  path='/extend/A/weather-benchmark/10m_U_component_wind/wind_prediction/picture',
                  out_len = 10):
    path = path + '/valid' + str(idx)
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    pred_path = path+'/'+'pred'

    if os.path.exists(pred_path):
        pass
    else:
        os.mkdir(pred_path)
    for idx_t in range(out_len):
        tmp = recons[idx_t]
        data = np.reshape(tmp, newshape=[101, 73])
        data = data*(max-min)+min
        time = y_data_time[idx_t]
        new_data = arr2ncl(arr_data=data, data_time=time, param=param)
        plt.figure(figsize=(10, 10))
        new_data.plot(vmin=-25, vmax=25,cmap='RdBu_r')
        # new_data.plot(cmap='RdBu_r')
        plt.savefig(pred_path+ '/' +str(time) + '.jpg')
        plt.close()


def arr2ncl(arr_data, data_time, param):
    lon = param['lon']
    lat = param['lat']
    data = arr_data
    t = data_time
    new_data = xr.DataArray(data,
                            dims={'lat': lat, 'lon': lon},
                            coords={"lat": lat, "lon": lon, 'time': t})
    new_data.attrs['units'] = 'm*s-1'
    new_data.attrs['long_name'] = 'Wind in y/longgitude-direction at 10m height'
    return new_data

def evaluate(idx, y, pred, length, valid_min, valid_max, path, total_mse, plot = True):
    mse_array = []
    psnr_array = []
    ssim_array = []
    bias_array = []
    for i in range(length):
        frame_y_i = y[:,i,:,:,:].reshape(101, 73)
        frame_pred_i = pred[:,i,:,:,:].reshape(101, 73)
        frame_y_i_recon = frame_y_i*(valid_max - valid_min) + valid_min
        frame_pred_i_recon = frame_pred_i * (valid_max - valid_min) + valid_min
        mean_bias = abs(frame_pred_i_recon - frame_y_i_recon).mean()
        mse = compare_mse(frame_y_i, frame_pred_i)
        psnr = compare_psnr(frame_y_i, frame_pred_i)
        ssim = compare_ssim(frame_y_i,frame_pred_i)

        mse_array.append(mse)
        psnr_array.append(psnr)
        ssim_array.append(ssim)
        bias_array.append(mean_bias)
    if plot == True:
        path = path + '/valid' + str(idx)
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)
        evaluation_path = path + '/' + 'evaluation.txt'
        file_writer = open(evaluation_path, mode='w')
        file_writer.write('Total_mse:' + str(total_mse) + ':\n')
        for i in range(length):
            file_writer.write('Frame_'+ str(i)+':\n')
            file_writer.write('mse:'+str(mse_array[i])+'; ')
            file_writer.write('psnr:'+str(psnr_array[i])+'; ')
            file_writer.write('ssim:'+str(ssim_array[i]) + '; ')
            file_writer.write('mean_bias:' + str(bias_array[i]))
            file_writer.write('\n')
        file_writer.close()
    else:
        mse_array = np.array(mse_array)
        psnr_array = np.array(psnr_array)
        ssim_array = np.array(ssim_array)
        bias_array = np.array(bias_array)
        return mse_array, psnr_array, ssim_array, bias_array

def plot(idx,ds,x_data_time,y_data_time,y,pred_image,total_mse,path,in_len,out_len,param,min,max):
    save_x_img(idx, ds, x_data_time=x_data_time[0], label='input',
                    path=path, in_len=in_len, out_len=out_len)
    save_x_img(idx, ds, x_data_time=y_data_time[0], label='output',
                    path=path, in_len=in_len, out_len=out_len)
    MNGO_save2jpg(idx=idx, recons=pred_image[0, :, :, :, :], y_data_time=y_data_time[0],
                       param=param, min=min, max=max, path=path,
                       out_len=out_len)
    evaluate(idx=idx, y= y, pred=pred_image, length=out_len, valid_min=min,
                  valid_max=max, path=path, total_mse=total_mse, plot=True)

