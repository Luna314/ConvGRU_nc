import torch
import torch.optim as optim
import High_deg_ConvGRU_getdata as datagenerator
import High_deg_ConvGRU_utils as util
import High_deg_ConvGRU_Encoder_Decoder_model as standard_gru
import os
import math
from tensorboardX import SummaryWriter
import numpy as np
from torch.autograd import Variable


def began_train(paras, start_iter=0):
    in_len = 5
    out_len = 10
    bs = 15
    lr = 0.0001
    if paras is not None:
        model = standard_gru.multilayers_Painter_model(out_length=out_len, in_length=in_len).cuda()
        save_model = torch.load(
            '/extend/A/weather-benchmark/10m_U_component_wind/multilayers_painters_test/model/standard_GRU_model/multilayers_grus_model4-ckpt.pth')
        model.load_state_dict(save_model['model_state_dict'])
    else:
        model = standard_gru.multilayers_Painter_model(out_length=out_len, in_length=in_len).cuda()
        print('current_model: Standard_ConvGRU')
        print('start pred,lr:', lr)
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=lr)
    train_data_generator = datagenerator.wind_data_generator(mode='train')
    n_sample = train_data_generator.train_ds_samples
    best_mse = math.inf
    for epoch in range(start_iter, 20):
        print(int((n_sample - (in_len + out_len) * train_data_generator.gap) / bs))
        for j in range(int((n_sample - (in_len + out_len) * train_data_generator.gap) / bs)):
            train_x, train_y, \
            x_time, y_time = train_data_generator.get_train_batch(input_length=in_len,
                                                                  output_length=out_len,
                                                                  batch_size=bs, item=j)
            train_x = Variable(torch.from_numpy(train_x).float().cuda(), requires_grad=True)
            train_y = Variable(torch.from_numpy(train_y).float().cuda(), requires_grad=True)
            train_x = train_x.reshape(bs, in_len, 1, 101, 73)
            train_y = train_y.reshape(bs, out_len, 1, 101, 73)

            optimizer.zero_grad()
            out = model.forward(train_x)
            out = out.cuda()

            recons_loss = torch.nn.MSELoss()
            loss = recons_loss(out, train_y)
            loss.backward()
            optimizer.step()
            if (j+1) % 50 == 0:
                print('Train epoch is:', epoch, 'curren iter:', j,
                      'loss:',float(loss.cpu().data.numpy()) )
                print('current_best_loss:',best_mse)
            if loss < best_mse:
                best_mse = float(loss.cpu().data.numpy())
                print('Train epoch is:', epoch, 'curren iter:', j,
                      'best_loss:', best_mse)
                ckpt_path = os.path.join('/mnt/A/era5/U_wind_500hPa/model',
                                         'Standard_ConvGRU_best_model-ckpt.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, ckpt_path)
                print('best_model_saved')
        ckpt_path = os.path.join('/mnt/A/era5/U_wind_500hPa/model',
                                 'Standard_ConvGRU_' + str(epoch) + '-ckpt.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
        }, ckpt_path)
        print('model_saved')

def begin_test(saved_model,iter = 0):
    in_len = 5
    out_len = 10
    bs = 1
    model_name = 'Standard_ConvGRU_test1'
    evalute_path = '/mnt/A/era5/U_wind_500hPa/pictures/' + model_name
    if os.path.exists(evalute_path):
        pass
    else:
        os.mkdir(evalute_path)
    model = standard_gru.multilayers_Painter_model(out_length=out_len, in_length=in_len).cuda()
    save_model = torch.load(saved_model)
    model.load_state_dict(save_model['model_state_dict'])
    model.eval()
    valid_best_mse = math.inf
    valid_worst_mse = 0
    valid_data_generator = datagenerator.wind_data_generator(mode='valid')
    valid_samples = valid_data_generator.valid_samples
    total_mse = np.zeros(shape=[10])
    total_psnr = np.zeros(shape=[10])
    total_ssim = np.zeros(shape=[10])
    total_bias = np.zeros(shape=[10])

    best_path = evalute_path + 'best_mse'
    if os.path.exists(best_path):
        pass
    else:
        os.mkdir(best_path)

    worst_path = evalute_path + 'worst_mse'
    if os.path.exists(worst_path):
        pass
    else:
        os.mkdir(worst_path)
        # valid_samples - (in_len + out_len) * valid_data_generator.gap
    for j in range(valid_samples - (in_len + out_len) * valid_data_generator.gap):
        print('current iter:', j)
        valid_x, valid_y, valid_x_time, valid_y_time = valid_data_generator.get_valid_batch(input_length=in_len,
                                                                             output_length=out_len,
                                                                             batch_size=bs, item=j)
        valid_x = torch.from_numpy(valid_x).float().cuda()
        valid_y = torch.from_numpy(valid_y).float().cuda()
        valid_x = valid_x.reshape(bs, in_len, 1, 101, 73)
        valid_y = valid_y.reshape(bs, out_len, 1, 101, 73)
        valid_out = model.forward(valid_x)
        criterion = torch.nn.MSELoss()
        loss = criterion(valid_out, valid_y)
        print('current valid loss is:', float(loss.cpu().data.numpy()))
        valid_out = valid_out.reshape(bs, out_len, 1, 101, 73)
        pred_image = valid_out.contiguous().cpu().detach().numpy()
        y = valid_y.contiguous().cpu().detach().numpy()
        current_mse, current_psnr, current_ssim, current_bias = util.evaluate(idx=j, y=y, pred=pred_image, length=out_len,
                                                                            valid_min=valid_data_generator.valid_data_min,
                              valid_max=valid_data_generator.valid_data_max, path=evalute_path, total_mse=None, plot = False)
        total_mse = total_mse + current_mse
        total_psnr = total_psnr + current_psnr
        total_ssim = total_ssim + current_ssim
        total_bias = total_bias + current_bias
        if loss < valid_best_mse:
            valid_best_mse = loss
            best_mse = valid_best_mse.contiguous().cpu().detach().numpy()
            best_idx = j
            best_x_data_time = valid_x_time
            best_y_data_time = valid_y_time
            best_pred_images = pred_image
            best_y = valid_y.contiguous().cpu().detach().numpy()
        elif loss > valid_worst_mse:
            valid_worst_mse = loss
            worst_mse = valid_worst_mse.contiguous().cpu().detach().numpy()
            worst_idx = j
            worst_x_data_time = valid_x_time
            worst_y_data_time = valid_y_time
            worst_pred_images = pred_image
            worst_y = valid_y.contiguous().cpu().detach().numpy()
        if j == 0:
            util.save_x_img(j+1, valid_data_generator.valid_ds, x_data_time=valid_x_time[0], label='input',
                            path=evalute_path, in_len=in_len, out_len=out_len)
            util.save_x_img(j+1, valid_data_generator.valid_ds, x_data_time=valid_y_time[0], label='output',
                            path=evalute_path, in_len=in_len, out_len=out_len)
            util.MNGO_save2jpg(idx=j+1, recons=pred_image[0, :, :, :, :], y_data_time=valid_y_time[0],
                               param=valid_data_generator.param,
                               min=valid_data_generator.valid_data_min,
                               max=valid_data_generator.valid_data_max, path=evalute_path,
                               out_len=out_len)
            util.evaluate(idx=j, y=y, pred=pred_image, length=out_len,
                          valid_min=valid_data_generator.valid_data_min,
                          valid_max=valid_data_generator.valid_data_max,
                          path=evalute_path, total_mse=best_mse)

    util.plot(idx=best_idx, ds=valid_data_generator.valid_ds, x_data_time=best_x_data_time,
              y_data_time=best_y_data_time,y = best_y,
              pred_image=best_pred_images,total_mse = best_mse, path=best_path,
              in_len = in_len, out_len=out_len,param=valid_data_generator.param,
              min = valid_data_generator.valid_data_min,
              max = valid_data_generator.valid_data_max)
    util.plot(idx=worst_idx, ds=valid_data_generator.valid_ds, x_data_time=worst_x_data_time,
              y_data_time=worst_y_data_time,y = worst_y,
              pred_image=worst_pred_images, total_mse=worst_mse, path=worst_path,
              in_len=in_len, out_len=out_len, param=valid_data_generator.param,
              min=valid_data_generator.valid_data_min,
              max=valid_data_generator.valid_data_max)
    evalution_path = evalute_path + 'evaluation.txt'
    file_writer = open(evalution_path, mode='w')
    mean_mse = total_mse/(j+1)
    mean_psnr = total_psnr/(j+1)
    mean_ssim = total_ssim/(j+1)
    mean_bias = total_bias / (j + 1)
    file_writer.write('Total_mse:' + str(np.sum(total_mse)) + ':\n')
    file_writer.write('mean_psnr:' + str(np.mean(mean_psnr)) + ':\n')
    file_writer.write('mean_ssim:' + str(np.mean(mean_ssim)) + ':\n')
    file_writer.write('mean_bias:' + str(np.mean(mean_bias)) + ':\n')
    for i in range(out_len):
        file_writer.write('Frame_' + str(i) + ':\n')
        file_writer.write('mse:' + str(mean_mse[i]) + '; ')
        file_writer.write('psnr:' + str(mean_psnr[i]) + '; ')
        file_writer.write('ssim:' + str(mean_ssim[i]) + '; ')
        file_writer.write('mean_bias:' + str(mean_bias[i]))
        file_writer.write('\n')
    file_writer.close()


if __name__ == '__main__':
    # began_train(paras=None, start_iter=0)
    begin_test(saved_model=
         '/mnt/A/era5/U_wind_500hPa/model/Standard_ConvGRU_best_model-ckpt.pth')