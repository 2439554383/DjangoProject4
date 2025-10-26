import time

import requests

# ssl = 'uLeQd9rHTh0GQG7RDiSzF8gJjvpVKxbFPy411f7djE9JC2P8eefOxLaO/BdnbmMejYFjN6NYDE6F2H+N6IaPXCRVpj89SPeY4yTbE4QIwg0DczGzxU0VE+cK4DHKa/uIrlCNL5tdJPL5hJ+NHFA3G6jNw8uhfB4g/rjeF+W/gmCkNWmXQ+TKzJOh7M5+jfcXc3/ew1Us5CM8Rui/mlpMMAQX8C/a+fEQpi1QguYDnGtWr+H7A5MZtaiMAn+heCfr1U4EgcBemc2/ehjOp3tELRKrOMQFS3dVKP8EWBTWbvIqlVZlxV0xM8LjyYhKbVFUgMb9Bg9XUC+IyJLl4lVdsXYQuyXT1ncT2nnzrhz3NPwFx/FGAt/Nw6jHIp7b+3uMwkdNaL8WHNPuA/FKbbg5AoYdF16+FNdid15e3bEJwiPl9Wc4/Pbx0/o66HxSLiw/7w60AcWvRET0DQJXq8hVVA=='
# url = 'http://139.196.235.10/AppEn.php?appid=12345678&m=3b10dc6194ecc6add629061e45790a68'
# mutualkey = '03a9f86fc3b6278af71785dd98ec3db7'
# date = ''
# api = 'getdate.ic'
# appsafecode=''
# md5 = ''
#
# post_url = f'{url}&api={api}&BSphpSeSsL={ssl}&date={date}&md5={md5}&mutualkey={mutualkey}&appsafecode={appsafecode}'
# print(post_url)
#
# response = requests.get(post_url)
# print(response)

ssl = 'uLeQd9rHTh0GQG7RDiSzF8gJjvpVKxbFPy411f7djE9JC2P8eefOxLaO/BdnbmMejYFjN6NYDE6F2H+N6IaPXCRVpj89SPeY4yTbE4QIwg0DczGzxU0VE+cK4DHKa/uIrlCNL5tdJPL5hJ+NHFA3G6jNw8uhfB4g/rjeF+W/gmCkNWmXQ+TKzJOh7M5+jfcXc3/ew1Us5CM8Rui/mlpMMAQX8C/a+fEQpi1QguYDnGtWr+H7A5MZtaiMAn+heCfr1U4EgcBemc2/ehjOp3tELRKrOMQFS3dVKP8EWBTWbvIqlVZlxV0xM8LjyYhKbVFUgMb9Bg9XUC+IyJLl4lVdsXYQuyXT1ncT2nnzrhz3NPwFx/FGAt/Nw6jHIp7b+3uMwkdNaL8WHNPuA/FKbbg5AoYdF16+FNdid15e3bEJwiPl9Wc4/Pbx0/o66HxSLiw/7w60AcWvRET0DQJXq8hVVA=='
url = 'http://139.196.235.10/AppEn.php?appid=12345678&m=3b10dc6194ecc6add629061e45790a68'
mutualkey = '03a9f86fc3b6278af71785dd98ec3db7'
date = str(int(time.time()))
api = 'login.ic'
appsafecode=''
md5 = ''
icid="xvXYDKr6pQJMQS8YFP1Y6NlrC7dHlsuuZLX1"
icpwd=""
key=""
maxoror='10'
post_url = f'{url}&api={api}&BSphpSeSsL={ssl}&date={date}&mutualkey={mutualkey}&appsafecode={appsafecode}&md5={md5}&icid={icid}&icpwd={icpwd}&key={key}&maxoror={maxoror}'
print(post_url)

response = requests.get(post_url)
print(response.text)