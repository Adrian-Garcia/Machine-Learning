	f��!`�?f��!`�?!f��!`�?	���ȿ�@���ȿ�@!���ȿ�@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$f��!`�?�4���?A�����?Y�0�:9C�?*	�p=
�S]@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat5�|�ݮ�?!���C@).��H٢?1�o:�a?@:Preprocessing2F
Iterator::Modelʥ��$�?!s�KM�D@)�(��/�?1���v7@:Preprocessing2U
Iterator::Model::ParallelMapV2������?!8���e2@)������?18���e2@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�v��?!�X��s)@)�[z4Ճ?1#�� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���SV�?!\��� @)���SV�?1\��� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)����u�?!�b��M@)ŪA�۽|?1'5%�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice@�z��{u?!tk�q]�@)@�z��{u?1tk�q]�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t10.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9���ȿ�@I�7t�W@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�4���?�4���?!�4���?      ��!       "      ��!       *      ��!       2	�����?�����?!�����?:      ��!       B      ��!       J	�0�:9C�?�0�:9C�?!�0�:9C�?R      ��!       Z	�0�:9C�?�0�:9C�?!�0�:9C�?JCPU_ONLYY���ȿ�@b q�7t�W@