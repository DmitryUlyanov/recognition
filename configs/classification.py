def update_defaults(parser):

    parser.add('--model', type=str, default="resnet_classification", help='')
    parser.add('--dataloader', type=str, default="classification", help='')
    parser.add('--runner', type=str, default="multiclass_separate", help='')

    parser.add('--save_frequency',  type=int, default=5, help='manual seed')

    parser.add('--num_epochs', type=int, default=100, help='manual seed')
    parser.add('--patience', type=int, default=10)
    

    parser.add('--optimizer', default="SGD")
    parser.add('--optimizer_args', default="lr=3e-3^momentum=0.9", type=str, help='separated with "^" list of args i.e. "lr=1e-3^betas=(0.5,0.9)"')

    parser.add('--image_size', default=380, type=int)
   


    #TO EDIT
    parser.add('--experiments_dir', type=str, default="extensions/projectname/data/experiments", help='')
    parser.add('--extension', type=str, default='tianchi')

    parser.add('--splits_dir', default="data/splits", type=str)
    parser.add('--target_columns', default="collar_design_labels,neckline_design_labels,skirt_length_labels,sleeve_length_labels,neck_design_labels,coat_length_labels,lapel_design_labels,pant_length_labels", type=str)

