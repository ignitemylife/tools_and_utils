data = dict(
)
model=dict(name='SafetyembActionModelV1', feat_dim=1024, d_model=512, n_head=8, n_layers=3, emb_dim=64)
learn = dict(
    total_epochs=50,
    resume=dict(resume_from=None, resume_optimizer=False, ),
    loss=dict(name='default'),
    optimizer=dict(name='AdamW', lr=0.0001),
    lr_scheduler=dict(name='MultiStepLR', milestones=[1000]),
    warmup_params=dict(warmup_iters=1_0000, warmup_factor=0.1),
    ckpt_cfg=dict(save_interval=1, validate=False),
    validate_interval=-1,  # if equal to -1, validate if and only if learn.ckpt_cfg.validation is True when saving ckpts
    print_freq=5,
    log_config=dict(write_tb=True, write_txt=True, print_interval=10),
    amp=False,
)
dist_params=dict()

resume_from=None
load_from = None
work_dir = 'output/demo'
log_level = 'INFO'
