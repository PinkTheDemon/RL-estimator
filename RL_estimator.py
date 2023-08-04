import dynamics as dyn





if __name__ == '__main__' : 
    (wlist,vlist) = dyn.gen_noise(1,10)
    print(wlist,'\n',vlist) 
    x = [1,2]
    x_next,y = dyn.step(x,wlist[0],vlist[0])
    print(x,y,x_next) 
    x_next,y = dyn.step(x,wlist[1],vlist[1]) 
    print(y) # 测试跨文件函数是否正确执行并得到相同结果——无误