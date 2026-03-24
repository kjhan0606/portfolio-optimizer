from pythonforandroid.recipes.numpy import NumPyRecipe


class NumPyRecipe126(NumPyRecipe):
    version = '1.26.4'
    url = 'https://github.com/numpy/numpy/releases/download/v{version}/numpy-{version}.tar.gz'


recipe = NumPyRecipe126()
