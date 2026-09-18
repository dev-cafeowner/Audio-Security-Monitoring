# Zybo Z7-20 SSM2603 audio codec pins.
# Source: Digilent Zybo-Z7-Master.xdc.

set_property -dict { PACKAGE_PIN R19 IOSTANDARD LVCMOS33 } [get_ports {ac_bclk}]
set_property -dict { PACKAGE_PIN R17 IOSTANDARD LVCMOS33 } [get_ports {ac_mclk}]
set_property -dict { PACKAGE_PIN P18 IOSTANDARD LVCMOS33 } [get_ports {ac_muten}]
set_property -dict { PACKAGE_PIN R18 IOSTANDARD LVCMOS33 } [get_ports {ac_pbdat}]
set_property -dict { PACKAGE_PIN T19 IOSTANDARD LVCMOS33 } [get_ports {ac_pblrc}]
set_property -dict { PACKAGE_PIN R16 IOSTANDARD LVCMOS33 } [get_ports {ac_recdat}]
set_property -dict { PACKAGE_PIN Y18 IOSTANDARD LVCMOS33 } [get_ports {ac_reclrc}]
set_property -dict { PACKAGE_PIN N18 IOSTANDARD LVCMOS33 } [get_ports {audio_iic_scl_io}]
set_property -dict { PACKAGE_PIN N17 IOSTANDARD LVCMOS33 } [get_ports {audio_iic_sda_io}]

# The SSM2603 control bus is asynchronous to the AXI fabric clock. AXI IIC
# implements the serial protocol and clock stretching, so there is no useful
# board-level synchronous timing contract for these two bidirectional pins.
set_false_path -from [get_ports {audio_iic_scl_io audio_iic_sda_io}]
set_false_path -to   [get_ports {audio_iic_scl_io audio_iic_sda_io}]
