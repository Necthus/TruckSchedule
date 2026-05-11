Clear-Content chihaya.log


$dispatch_method_list = @('Fastest','Follow')
$reposition_method_list = @('Urgent','Retrace')

foreach ($dispatch_method in $dispatch_method_list) {
    foreach ($reposition_method in $reposition_method_list) {
        python ./main_process.py --dispatch_method=$dispatch_method --reposition_method=$reposition_method >> chihaya.log
    }
}

