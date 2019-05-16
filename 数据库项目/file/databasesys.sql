/*
Navicat MySQL Data Transfer

Source Server         : localhost3306
Source Server Version : 50520
Source Host           : localhost:3306
Source Database       : databasesys

Target Server Type    : MYSQL
Target Server Version : 50520
File Encoding         : 65001

Date: 2019-05-16 19:14:46
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for `login`
-- ----------------------------
DROP TABLE IF EXISTS `login`;
CREATE TABLE `login` (
  `ID` int(11) NOT NULL,
  `UserName` varchar(20) NOT NULL,
  `Passwd` varchar(20) DEFAULT NULL,
  `Prior` enum('admin','user') NOT NULL,
  `Email` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`ID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- ----------------------------
-- Records of login
-- ----------------------------
INSERT INTO `login` VALUES ('1', 'Adam', '12345', 'admin', '123@qq.com');
