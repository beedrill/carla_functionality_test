import  math
def continued_fraction_remain(n, iters):
  if n == 0:
    return '...]'
  if iters == 0:
    return '...]'
  t = math.floor(1/n)
  n = 1/n -t
  return f'{t},' + continued_fraction_remain(n, iters-1)
def continued_fraction(n, iters):
  return f'[{math.floor(n)};' + continued_fraction_remain(n-math.floor(n), iters-1)

print(continued_fraction(7/4, 3))
print(continued_fraction(math.e, 12))
print(continued_fraction(math.pi+3279, 1))