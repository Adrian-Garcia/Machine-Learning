�	��ƽ��?��ƽ��?!��ƽ��?	0�[�~�,@0�[�~�,@!0�[�~�,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��ƽ��?�m�2�?A�쟧��?YI.�!���?*	y�&1�R@2U
Iterator::Model::ParallelMapV2��n,(�?!�Z�+,A@)��n,(�?1�Z�+,A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat{���Η?!S�O]c?@)���Փ?1�Ȝ�E':@:Preprocessing2F
Iterator::Model\����o�?!N��;i�F@)�]�)ʥ�?1	���D'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap:!t�%�?!8@���61@)�+�j��?1j�S[�&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceB���Dr?!e���@)B���Dr?1e���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipFaE|�?!�Z/ĖK@)�=$|�op?1d�Y�f�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�f����o?!W �]�@)�f����o?1W �]�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 14.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t30.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9/�[�~�,@I��� �gU@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�m�2�?�m�2�?!�m�2�?      ��!       "      ��!       *      ��!       2	�쟧��?�쟧��?!�쟧��?:      ��!       B      ��!       J	I.�!���?I.�!���?!I.�!���?R      ��!       Z	I.�!���?I.�!���?!I.�!���?JCPU_ONLYY/�[�~�,@b q��� �gU@Y      Y@q��k��PV@"�	
both�Your program is MODERATELY input-bound because 14.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nohigh"t30.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb�89.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 