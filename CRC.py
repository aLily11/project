

def XOR(str1, str2):  # 实现模2减法
    ans = ''
    if str1[0] == '0':
        return '0', str1[1:]
    else:
        for i in range(len(str1)):
            if (str1[i] == '0' and str2[i] == '0'):
                ans = ans + '0'
            elif (str1[i] == '1' and str2[i] == '1'):
                ans = ans + '0'
            else:
                ans = ans + '1'
    return '1', ans[1:]

def CRC_Encoding(str1,str2):    #CRC编码
    lenght = len(str2)
    str3 = str1 + '0'*(lenght-1)
    ans = ''
    yus = str3[0:lenght]
    for i in range(len(str1)):
        str4,yus = XOR(yus, str2)
        ans = ans+str4
        if i == len(str1)-1:
            break
        else:
            yus = yus+str3[i+lenght]
    ans = str1 + yus
    return ans

def CRC_Decoding(str1,str2):    #CRC解码
    lenght = len(str2)
    str3 = str1 + '0'*(lenght-1)
    ans = ''
    yus = str3[0:lenght]
    for i in range(len(str1)):
        str4,yus = XOR(yus, str2)
        ans = ans+str4
        if i == len(str1)-1:
            break
        else:
            yus = yus+str3[i+lenght]
    return yus == '0'*len(yus)