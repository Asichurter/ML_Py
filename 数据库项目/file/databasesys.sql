/*
Navicat MySQL Data Transfer

Source Server         : localhost3306
Source Server Version : 50520
Source Host           : localhost:3306
Source Database       : databasesys

Target Server Type    : MYSQL
Target Server Version : 50520
File Encoding         : 65001

Date: 2019-05-18 00:25:21
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for `login`
-- ----------------------------
DROP TABLE IF EXISTS `login`;
CREATE TABLE `login` (
  `ID` varchar(5) NOT NULL,
  `UserName` varchar(20) NOT NULL,
  `Passwd` varchar(20) DEFAULT NULL,
  `Prior` enum('admin','user') NOT NULL,
  `Email` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`ID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- ----------------------------
-- Records of login
-- ----------------------------
INSERT INTO `login` VALUES ('00001', '1', '1', 'admin', '1');
INSERT INTO `login` VALUES ('10000', 'zhangkai', '11111', 'admin', '11111@qq.com');
INSERT INTO `login` VALUES ('10001', 'a', '12345', 'user', 'a@qq.com');

-- ----------------------------
-- Table structure for `studentinfo`
-- ----------------------------
DROP TABLE IF EXISTS `studentinfo`;
CREATE TABLE `studentinfo` (
  `ID` varchar(5) NOT NULL,
  `Department` varchar(10) DEFAULT NULL,
  `Majority` varchar(10) DEFAULT NULL,
  `Sex` enum('Female','Male') DEFAULT NULL,
  `Address` varchar(20) DEFAULT NULL,
  `Telephone` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`ID`),
  CONSTRAINT `ID_ref` FOREIGN KEY (`ID`) REFERENCES `login` (`ID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- ----------------------------
-- Records of studentinfo
-- ----------------------------
INSERT INTO `studentinfo` VALUES ('00001', 'Software', null, null, null, null);
INSERT INTO `studentinfo` VALUES ('10000', 'Software', 'Software', 'Male', 'Beijing', '110');
INSERT INTO `studentinfo` VALUES ('10001', null, null, null, null, null);

-- ----------------------------
-- Table structure for `takes`
-- ----------------------------
DROP TABLE IF EXISTS `takes`;
CREATE TABLE `takes` (
  `ID` varchar(5) NOT NULL,
  `Class` varchar(20) NOT NULL,
  `Grade` decimal(4,1) NOT NULL,
  PRIMARY KEY (`ID`,`Class`),
  CONSTRAINT `id_ref_info` FOREIGN KEY (`ID`) REFERENCES `studentinfo` (`ID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- ----------------------------
-- Records of takes
-- ----------------------------
INSERT INTO `takes` VALUES ('10000', 'C', '100.0');
INSERT INTO `takes` VALUES ('10000', 'Caculars', '88.0');
INSERT INTO `takes` VALUES ('10000', 'Database', '90.0');
INSERT INTO `takes` VALUES ('10000', 'English', '91.0');
INSERT INTO `takes` VALUES ('10000', 'History', '15.0');
INSERT INTO `takes` VALUES ('10000', 'Java', '92.0');
INSERT INTO `takes` VALUES ('10000', 'Operating System', '97.0');
