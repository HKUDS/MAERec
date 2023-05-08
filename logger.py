import datetime

logmsg = ''
timemark = dict()
saveDefault = False

def log(msg, save=None, oneline=False, bold=False):
    global logmsg
    global saveDefault
    time = datetime.datetime.now()
    tem = '%s: %s' % (time, msg)

    if save != None:
        if save:
            logmsg += tem + '\n'
    elif saveDefault:
        logmsg += tem + '\n'

    if bold:
        tem = '\033[1m' + tem + '\033[0m'

    if oneline:
        print(tem, end='\r')
    else:
        print(tem)

if __name__ == '__main__':
    log('')
