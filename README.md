# condnet

## 실행방법    
1. Windows 11, Python 3.9 에서 테스트 되었습니다.    
2.     conda install theano     
3.     conda install matplotlib    
4.     conda install numpy==1.19.5    
5.     pip install pickle-mixin    
6.     pip install nose    
10. 모델은 `condconv` 에 선언되어 있습니다.(아마)     
    
    

## 보고된 이슈    
    
### dependencies
WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
C:\Users\97dnd\Desktop\condnet-master\theano_tools\__init__.py:24: UserWarning: top level import of theano_tools may not import what you expect!
  warnings.warn("top level import of theano_tools may not import what you expect!")
Traceback (most recent call last):
  File "C:\Users\97dnd\Desktop\condnet-master\condconv.py", line 7, in <module>
    from theano_tools.deep import ConvLayer, HiddenLayer, StackModel, Maxpool, relu, shared
  File "C:\Users\97dnd\Desktop\condnet-master\theano_tools\deep.py", line 7, in <module>
    import theano.gpuarray as gpuarray
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gpuarray\__init__.py", line 33, in <module>
    from . import fft, dnn, opt, extra_ops, multinomial, reduction, sort, rng_mrg, ctc
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gpuarray\fft.py", line 14, in <module>
    from .opt import register_opt, op_lifter, register_opt2
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gpuarray\opt.py", line 2812, in <module>
    from .dnn import (local_abstractconv_cudnn,
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gpuarray\dnn.py", line 339, in <module>
    handle_type = CUDNNDataType('cudnnHandle_t', 'cudnnDestroy')
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gpuarray\dnn.py", line 259, in CUDNNDataType
    version=version(raises=False))
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gpuarray\dnn.py", line 319, in version
    if not dnn_present():
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gpuarray\dnn.py", line 209, in dnn_present
    dnn_present.avail, dnn_present.msg = _dnn_check_version()
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gpuarray\dnn.py", line 180, in _dnn_check_version
    v = version()
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gpuarray\dnn.py", line 328, in version
    f = theano.function([], DnnVersion()(),
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\compile\function.py", line 306, in function
    fn = pfunc(params=inputs,
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\compile\pfunc.py", line 483, in pfunc
    return orig_function(inputs, cloned_outputs, mode,
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\compile\function_module.py", line 1841, in orig_function
    fn = m.create(defaults)
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\compile\function_module.py", line 1714, in create
    _fn, _i, _o = self.linker.make_thunk(
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gof\link.py", line 697, in make_thunk
    return self.make_all(input_storage=input_storage,
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gof\vm.py", line 1087, in make_all
    thunks.append(node.op.make_thunk(node,
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gof\op.py", line 954, in make_thunk
    return self.make_c_thunk(node, storage_map, compute_map,
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gof\op.py", line 857, in make_c_thunk
    outputs = cl.make_thunk(input_storage=node_input_storage,
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gof\cc.py", line 1215, in make_thunk
    cthunk, module, in_storage, out_storage, error_storage = self.__compile__(
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gof\cc.py", line 1153, in __compile__
    thunk, module = self.cthunk_factory(error_storage,
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gof\cc.py", line 1623, in cthunk_factory
    module = get_module_cache().module_from_key(
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gof\cmodule.py", line 1189, in module_from_key
    module = lnk.compile_cmodule(location)
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gof\cc.py", line 1520, in compile_cmodule
    module = c_compiler.compile_str(
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gof\cmodule.py", line 2423, in compile_str
    return dlimport(lib_filename)
  File "C:\Users\97dnd\anaconda3\envs\theano\lib\site-packages\theano\gof\cmodule.py", line 317, in dlimport
    rval = __import__(module_name, {}, {}, [module_name])
ImportError: DLL load failed while importing mb723a79e77767d1dbdefdd68a32a62b84558aa21c10528d0101d296d860cddc8: 지정된 모듈을 찾을 수 없습니다.

Process finished with exit code 1
